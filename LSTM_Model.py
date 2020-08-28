#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug  5 2020

@author: Abhinesh Mishra
"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas import read_csv
import math
import io
from google.colab import files

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Find the Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
  y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
  return numpy.mean(numpy.absolute(y_pred - numpy.mean(y_true)))

# Create dataset Matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return numpy.array(dataX), numpy.array(dataY)

# Random seed configuration for reproducibility
numpy.random.seed(7)

#uploaded = files.upload()
# Making a list of missing value types
missing_values = ["n/a", "na", "--","-","."]
df1 = read_csv(io.BytesIO(uploaded['new_dos.csv']), usecols=['Num_of_DoS'], na_values=missing_values) 
print(df1.head(5))
dataset = df1

#dataset['protocol_type']= le.fit_transform(dataset['protocol_type']) 
#dataset['service']= le.fit_transform(dataset['service']) 
#dataset['flag']= le.fit_transform(dataset['flag']) 

# normalizing the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Splitting the data into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Setting the look back hyper-parameter
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshaping the values into X=t and Y=t+1
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Creating and fitting LSTM
model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=150, batch_size=5, verbose=2)

# Making the predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverting the predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print(‘The Train Score: %.2f RMSE' % (trainScore))
train_mad = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print(‘The Train MAD: %.2f MAD' % (train_mad))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print(‘The Test Score: %.2f RMSE' % (testScore))
test_mad = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print(‘The Test MAD: %.2f MAD' % (test_mad))

# Shifting the train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shifting the test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Plotting baseline and predictions
plt.figure(figsize=(20,6))
font = {'family' : 'normal',
        'weight' : 500,
        'size'   : 16}

plt.rc('font', **font)
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.xlabel('Days',fontweight=600)
plt.ylabel('Daily DoS Count',fontweight=600)
blue_patch = mpatches.Patch(color='blue', label='Original Plot')
orange_patch = mpatches.Patch(color='#FF9300', label='Train Prediction')
green_patch = mpatches.Patch(color='#01a982', label='Test Prediction')
plt.legend(handles=[blue_patch,orange_patch,green_patch])
plt.show()

