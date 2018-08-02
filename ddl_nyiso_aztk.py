"""
// Copyright (c) Microsoft Corporation. All rights reserved. 
// Licensed under the MIT license.

Distributed Training of RNN Models on an AZTK Spark Cluster

This notebook demonstrates energy consumption prediction using Distributed Keras on an AZTK Spark cluster. It uses distkeras package to train
a GRU model and an LSTM model in a distrbuted manner on the Spark cluster. The data used is the NYISO data which describes the hourly energy 
consumption of New York City. We use both the energy consumption data itself and weather data to make the prediction. To run this script, 
you can first create a folder named 'scripts' under the AZTK source code folder and place the script to the created folder. Then, you can 
submit a Spark job by executing the following command
  
                    aztk spark cluster submit --id <my_cluster_id> --name <my_job_id> scripts/ddl_nyiso_aztk.py 

This example requires the following packages
- distkeras
- keras
- tensorflow
"""

import os
import time
import pyspark

import datetime as dt
import numpy as np
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import *

from distkeras.evaluators import *
from distkeras.predictors import *
from distkeras.trainers import *
from distkeras.transformers import *
from distkeras.utils import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU

from pyspark.ml.feature import VectorAssembler

## Setup the pyspark environment

# First, setup the Spark variables. You can modify them according to your need, e.g. increasing num_executors to reduce training time.
application_name = 'Distributed Keras NYISO'
master = 'spark://' + os.environ.get('AZTK_MASTER_IP') + ':7077' 
num_processes = 2 
num_executors = 2 #4, 3, 2, 1

# This variable is derived from the number of cores and executors, and will be used to assign the number of model trainers.
num_workers = num_executors * num_processes

print('Number of desired executors: ' + str(num_executors))
print('Number of desired processes per executor: ' + str(num_processes))
print('Total number of workers: ' + str(num_workers))

# Use the DataBricks CSV reader, which has some nice functionality regarding invalid values.
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-csv_2.10:1.4.0 pyspark-shell'

# Modify the Spark configuration
conf = pyspark.SparkConf()
conf.set('spark.app.name', application_name)
conf.set('spark.master', master)
conf.set('spark.executor.cores', num_processes)
conf.set('spark.executor.instances', num_executors)
conf.set('spark.executor.memory', '2g')
conf.set('spark.locality.wait', '0')
conf.set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer');
conf.set('spark.local.dir', '/tmp/' + get_os_username() + '/dist-keras');

# Create the Spark context
sc = pyspark.SparkContext(conf=conf)
sqlc = pyspark.sql.SQLContext(sc)

## Define variables
LEN_SEQ_IN = 24
LEN_SEQ_OUT = 1
LEN_EXTRA_IN = LEN_SEQ_OUT
LEN_TEST_DATA = 120
N_UNITS = 128

BATCH_SIZE = 32
COM_WINDOW = 5
N_EPOCHS = 20

input_container = 'nyiso'
storage_account = 'publicdat'
storage_key = 'GNDru9kic+4jXle1roq4klFzwQ3Jy9CvsvMJl8o5b8W/DoX6v14PE8aPuX48V3r9yc3tuKP3NKCSUHqFobib4A=='
data_path = 'wasb://{}@{}.blob.core.windows.net/NYISO_data_1region_small.csv'.format(input_container, storage_account)

## Load data

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

# Load the input dataset
raw_df = sqlc.read.format('com.databricks.spark.csv') \
                  .options(header='true', inferSchema='true') \
                  .load(data_path)

# Convert data type and select columns
func = F.udf (lambda x: dt.datetime.strptime(x[:19], '%m/%d/%Y %H:%M:%S'), TimestampType())
raw_df = raw_df.withColumn('TimeStamp', func(F.col('TimeStamp')))
df = raw_df.select(['TimeStamp', 'Name', 'HourAvgLoad', 'temperature'])
df.show(3)

# Normalize the data
orig_min = df.select(F.min('temperature')).collect()[0][0]
orig_max = df.select(F.max('temperature')).collect()[0][0]
transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, \
                                o_min=orig_min, o_max=orig_max, \
                                input_col='temperature', \
                                output_col='NormTemp', \
                                is_vector=False)
df = transformer.transform(df)
orig_min = df.select(F.min('HourAvgLoad')).collect()[0][0]
orig_max = df.select(F.max('HourAvgLoad')).collect()[0][0]
transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, \
                                o_min=orig_min, o_max=orig_max, \
                                input_col='HourAvgLoad', \
                                output_col='NormLoad', \
                                is_vector=False)
df = transformer.transform(df)
df.show(3)

# Create input features and output targets
wSpec = Window.partitionBy('Name').orderBy('TimeStamp')
for n_lag in range(LEN_SEQ_IN, 0, -1):
    df = df.withColumn('NormLoad_lag'+str(n_lag), F.lag(F.col('NormLoad'), count = n_lag).over(wSpec))
for n_lag in range(1, LEN_SEQ_OUT+1):
    df = df.withColumn('NormLoad_next'+str(n_lag), F.lead(F.col('NormLoad'), count = n_lag).over(wSpec))
    df = df.withColumn('OrigNormLoad_next'+str(n_lag), F.lead(F.col('HourAvgLoad'), count = n_lag).over(wSpec))
    df = df.withColumn('NormTemp_next'+str(n_lag), F.lead(F.col('NormTemp'), count = n_lag).over(wSpec))

# Drop null values
df = df.na.drop()
df.show(1)

# Assemble all the features
features = ['NormLoad_lag'+str(n) for n in range(LEN_SEQ_IN, 0, -1)] + ['NormTemp_next'+str(n) for n in range(1, LEN_SEQ_OUT+1)]
vector_assembler = VectorAssembler(inputCols=features, outputCol='features')
df = vector_assembler.transform(df)
df.show(1)

# Reshape the vectors into the format that Keras requires
reshape_transformer = ReshapeTransformer('features', 'feature_matrix', (LEN_SEQ_IN+LEN_EXTRA_IN, 1))
df = reshape_transformer.transform(df)
df.show(1)

# Assemble all the target variables
targets = ['NormLoad_next'+str(n) for n in range(1, LEN_SEQ_OUT+1)]
vector_assembler = VectorAssembler(inputCols=targets, outputCol='labels')
df = vector_assembler.transform(df)

targets = ['OrigNormLoad_next'+str(n) for n in range(1, LEN_SEQ_OUT+1)]
vector_assembler = VectorAssembler(inputCols=targets, outputCol='labels2')
df = vector_assembler.transform(df)

df.show(1)

# Reshape the labels
reshape_transformer = ReshapeTransformer('labels', 'label_matrix', (LEN_SEQ_OUT, 1))
df = reshape_transformer.transform(df)
df.show(1)

# Partition the data into training and testing sets
df_train = df.limit(df.count() - LEN_TEST_DATA)
df2 = df
df2 = df2.orderBy('TimeStamp', ascending=False)
df_test = df2.limit(LEN_TEST_DATA).orderBy('TimeStamp', ascending=True)

df_train.show(1)
df_test.show(1)

# Select the desired columns to reduce network usage
df_train = df_train.select('features', 'feature_matrix', 'labels', 'labels2', 'label_matrix')
df_test = df_test.select('features', 'feature_matrix', 'labels', 'labels2', 'label_matrix')

# Repartition the data
df_train = df_train.repartition(num_workers)
df_test = df_test.repartition(num_workers)

# Cache the data
df_train.cache()
df_test.cache()

# Create the GRU network
gru_model = Sequential()
gru_model.add(GRU(N_UNITS, input_shape=(LEN_SEQ_IN+LEN_EXTRA_IN, 1))) 
gru_model.add(Dense(LEN_SEQ_OUT, activation='linear')) 
gru_model.summary()

# Fit the model
gru_trainer = ADAG(keras_model=gru_model, worker_optimizer='adagrad', loss='mean_squared_error',
                   num_workers=num_workers, batch_size=BATCH_SIZE, communication_window=COM_WINDOW, 
                   num_epoch=N_EPOCHS, features_col='feature_matrix', label_col='labels')
trained_gru_model = gru_trainer.train(df_train)

# gru_trainer = DynSGD(keras_model=gru_model, worker_optimizer='adagrad', loss='mean_squared_error',
#                    num_workers=num_workers, batch_size=BATCH_SIZE, communication_window=COM_WINDOW, 
#                    num_epoch=N_EPOCHS, features_col='feature_matrix', label_col='labels')
# trained_gru_model = gru_trainer.train(df_train)

print('Number of parameter updates ' + str(gru_trainer.parameter_server.num_updates))
print('Total training time in seconds ' + str(gru_trainer.get_training_time()))

predictor = ModelPredictor(keras_model=trained_gru_model, features_col='feature_matrix')
df_pred = predictor.predict(df_test.limit(24))
df_pred.show(24)

# Convert the values to the original range
inverse_transformer = MinMaxTransformer(n_min=orig_min, n_max=orig_max, \
                                        o_min=0.0, o_max=1.0, \
                                        input_col='prediction', \
                                        output_col='prediction2', \
                                        is_vector=True)
df_pred = inverse_transformer.transform(df_pred)
df_pred.show(24)

# Compute Mean-Absolute-Percentage Errors (MAPEs)
def get_MAPE(actual, pred): 
    """
    Compute the mean-absolute-percentage-error (MAPE)
    """
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    if mape == np.inf:
        mape = np.nan
    return mape

actual = df_pred.select('labels2').rdd.map(lambda x: list(x[0])).collect()
pred = df_pred.select('prediction2').rdd.map(lambda x: list(x[0])).collect()
get_MAPE(actual, pred)

# Create the LSTM network
lstm_model = Sequential()
lstm_model.add(LSTM(N_UNITS, input_shape=(LEN_SEQ_IN+LEN_EXTRA_IN, 1))) 
lstm_model.add(Dense(LEN_SEQ_OUT, activation='linear')) 
lstm_model.summary()

# Fit the model
lstm_trainer = ADAG(keras_model=lstm_model, worker_optimizer='adam', loss='mean_squared_error',
                    num_workers=num_workers, batch_size=BATCH_SIZE, communication_window=COM_WINDOW, 
                    num_epoch=N_EPOCHS, features_col='feature_matrix', label_col='labels')
trained_lstm_model = lstm_trainer.train(df_train)

print('Number of parameter updates ' + str(lstm_trainer.parameter_server.num_updates))
print('Total training time in seconds ' + str(lstm_trainer.get_training_time()))

predictor = ModelPredictor(keras_model=trained_lstm_model, features_col='feature_matrix')
df_pred = predictor.predict(df_test.limit(24))
df_pred.show(24)

# Convert the values to the original range
df_pred = inverse_transformer.transform(df_pred)
df_pred.show(24)

# Compute MAPE
actual = df_pred.select('labels2').rdd.map(lambda x: list(x[0])).collect()
pred = df_pred.select('prediction2').rdd.map(lambda x: list(x[0])).collect()
get_MAPE(actual, pred)
