'''
// Copyright (c) 2016 Joeri Hermans
// Copyright (c) Microsoft Corporation. All rights reserved. 
// Licensed under the GPLv3 license.

MNIST classification using Distributed Keras on an AZTK Spark cluster. This script is adapted from the  
dist-keras's example https://github.com/cerndb/dist-keras/blob/master/examples/mnist.py, written by Joeri 
Hermans (joeri.hermans@cern.ch). To run the script, you can first create a folder named 'scripts' under 
the AZTK source code folder and place it to the created folder. Then, you can submit a Spark job by 
executing the following command
  
      aztk spark cluster submit --id <my_cluster_id> --name <my_job_id> scripts/ddl_mnist.py 

This example requires the following packages
- distkeras 
- keras 
- tensorflow
'''

import os
import time
import pyspark

from pyspark.sql import SparkSession

from distkeras.evaluators import *
from distkeras.predictors import *
from distkeras.trainers import *
from distkeras.transformers import *
from distkeras.utils import *

from keras.layers.convolutional import *
from keras.layers.core import *
from keras.models import Sequential
from keras.optimizers import *

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler


## Setup the pyspark environment

# First, setup the Spark variables. You can modify them according to your need, e.g. increasing num_executors to reduce training time.
application_name = 'Distributed Keras MNIST'
master = 'spark://' + os.environ.get('AZTK_MASTER_IP') + ':7077' 
num_processes = 2 
num_executors = 4 #4, 3, 2, 1

# This variable is derived from the number of cores and executors, and will be used to assign the number of model trainers.
num_workers = num_executors * num_processes

print('Number of desired executors: ', num_executors)
print('Number of desired processes per executor: ', num_processes)
print('Total number of workers: ', num_workers)

# Use the DataBricks CSV reader, this has some nice functionality regarding invalid values.
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-csv_2.10:1.4.0 pyspark-shell'

# Modify the Spark configuration
conf = pyspark.SparkConf()
conf.set('spark.app.name', application_name)
conf.set('spark.master', master)
conf.set('spark.executor.cores', num_processes)
conf.set('spark.executor.instances', num_executors)
conf.set('spark.executor.memory', '2g')
conf.set('spark.locality.wait', '0')
conf.set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
conf.set('spark.local.dir', '/tmp/' + get_os_username() + '/dist-keras')

# Create the Spark context
sc = pyspark.SparkContext(conf=conf)
sqlc = pyspark.sql.SQLContext(sc)


## Load data

# Define variables
storage_account = 'publicdat'
storage_key = 'GNDru9kic+4jXle1roq4klFzwQ3Jy9CvsvMJl8o5b8W/DoX6v14PE8aPuX48V3r9yc3tuKP3NKCSUHqFobib4A=='
input_container = 'mnist'
path_train = 'wasb://{}@{}.blob.core.windows.net/mnist_train.csv'.format(input_container, storage_account)
path_test = 'wasb://{}@{}.blob.core.windows.net/mnist_test.csv'.format(input_container, storage_account)

# Attach the blob storage to the spark cluster  
def attach_storage_container(spark, account, key):
    config = spark._sc._jsc.hadoopConfiguration()
    setting = 'fs.azure.account.key.' + account + '.blob.core.windows.net'
    if not config.get(setting):
        config.set(setting, key)

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
attach_storage_container(spark, storage_account, storage_key)
time.sleep(4) 
	
# Read the training dataset
raw_dataset_train = sqlc.read.format('com.databricks.spark.csv') \
                          .options(header='true', inferSchema='true') \
                          .load(path_train)

# Read the testing dataset
raw_dataset_test = sqlc.read.format('com.databricks.spark.csv') \
                         .options(header='true', inferSchema='true') \
                         .load(path_test)


## Data transformation

# First, we extract the desired features from the raw dataset by constructing a list with all desired columns. This is identical for the test set.
features = raw_dataset_train.columns
features.remove('label')

# Next, we use Spark's VectorAssembler to assemble/create a vector of all the desired features.
# http://spark.apache.org/docs/latest/ml-features.html#vectorassembler
vector_assembler = VectorAssembler(inputCols=features, outputCol='features')
# This transformer will take all columns specified in features, and create an additional column 
# 'features' which will contain all the desired features aggregated into a single vector.
dataset_train = vector_assembler.transform(raw_dataset_train)
dataset_test = vector_assembler.transform(raw_dataset_test)

# Define the number of output classes
nb_classes = 10
encoder = OneHotTransformer(nb_classes, input_col='label', output_col='label_encoded')
dataset_train = encoder.transform(dataset_train)
dataset_test = encoder.transform(dataset_test)

# Allocate a MinMaxTransformer from Distributed Keras to normalize the features
# o_min -> original_minimum
# n_min -> new_minimum
transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, \
                                o_min=0.0, o_max=250.0, \
                                input_col='features', \
                                output_col='features_normalized')
# Transform the dataset
dataset_train = transformer.transform(dataset_train)
dataset_test = transformer.transform(dataset_test)

# Keras expects the vectors to be in a particular shape, we can reshape the vectors using Spark.
reshape_transformer = ReshapeTransformer('features_normalized', 'matrix', (28, 28, 1))
dataset_train = reshape_transformer.transform(dataset_train)
dataset_test = reshape_transformer.transform(dataset_test)

# Select the desired columns, this will reduce network usage.
dataset_train = dataset_train.select('features_normalized', 'matrix', 'label', 'label_encoded')
dataset_test = dataset_test.select('features_normalized', 'matrix', 'label', 'label_encoded')

# Keras expects DenseVectors
dense_transformer = DenseTransformer(input_col='features_normalized', output_col='features_normalized_dense')
dataset_train = dense_transformer.transform(dataset_train)
dataset_test = dense_transformer.transform(dataset_test)

# Repartition the training and test set
training_set = dataset_train.repartition(num_workers)
test_set = dataset_test.repartition(num_workers)

# Cache the data sets
training_set.cache()
test_set.cache()

# Precache the trainingset on the nodes using a simple count.
#print(training_set.count())


## Model and evaluation metric definition

# dimensions of the image
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = (img_rows, img_cols, 1)

# Construct the model
convnet = Sequential()
convnet.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                          padding='valid',
                          input_shape=input_shape))
convnet.add(Activation('relu'))
convnet.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
convnet.add(Activation('relu'))
convnet.add(MaxPooling2D(pool_size=pool_size))
convnet.add(Flatten())
convnet.add(Dense(225))
convnet.add(Activation('relu'))
convnet.add(Dense(nb_classes))
convnet.add(Activation('softmax'))

# Define the optimizer and the loss
optimizer_convnet = 'adam'
loss_convnet = 'categorical_crossentropy'

# Print the model structure
convnet.summary()

# Specify a procedure to evaluate the dataset in a distributed manner
def evaluate_accuracy(model, test_set, features='matrix'):
    evaluator = AccuracyEvaluator(prediction_col='prediction_index', label_col='label')
    predictor = ModelPredictor(keras_model=model, features_col=features)
    transformer = LabelIndexTransformer(output_dim=nb_classes)
    test_set = test_set.select(features, 'label')
    test_set = predictor.predict(test_set)
    test_set = transformer.transform(test_set)
    score = evaluator.evaluate(test_set)
    return score


## Model training and evaluation

# Use the ADAG optimizer. You can also use a SingleWorker to compare the performance with traditional non-distributed gradient descent.
trainer = ADAG(keras_model=convnet, worker_optimizer=optimizer_convnet, loss=loss_convnet,
               num_workers=num_workers, batch_size=16, communication_window=5, num_epoch=5,
               features_col='matrix', label_col='label_encoded')
trained_model = trainer.train(training_set)

print('----------------------------------------------------')
print('Training time: ', trainer.get_training_time())
print('Accuracy: ', evaluate_accuracy(trained_model, test_set))
print('Number of parameter server updates: ', trainer.parameter_server.num_updates)



