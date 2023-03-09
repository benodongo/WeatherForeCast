#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np  #numpy
import pandas as pd  #pandas
import tensorflow as tf
#date manipulation 
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sb


# ##Load the data from CSV

# In[2]:


data = pd.read_csv('D:\BENSON\Rabala\data.csv')
data.head() 


# # Data Preprocessing

# In[3]:


data.describe().T


# In[4]:


#data cleaning
for col in data.columns:

  # Checking if the column contains
  # any null values
  if data[col].isnull().sum() > 0:
    val = data[col].mean()
    data[col] = data[col].fillna(val)
    
data.isnull().sum().sum()


# In[5]:


#get average temperature
col = data.loc[: , "Tmax":"Tmin"]
data['average_temperature'] = col.mean(axis=1)


# In[6]:


#Data Visualisation
features = list(data.select_dtypes(include = np.number).columns)
features.remove('Date')
print(features)


# In[7]:


plt.subplots(figsize=(15,8))
 
for i, col in enumerate(features):
  plt.subplot(2,2, i + 1)
  sb.distplot(data[col])
plt.tight_layout()
plt.show()


# # Model Training

# In[ ]:


#Feature Scaling to  normalize temperature in the range 0 to 1.
dataset = data['average_temperature']
training_set = dataset.iloc[:,4:5].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(training_set_scaled)-n_past-n_future+1):
    x_train.append(training_set_scaled[i : i + n_past , 0])     
    y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout
# Fitting RNN to training set using Keras Callbacks. 
regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future,activation='linear'))
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
regressor.fit(x_train, y_train, epochs=500,batch_size=32 )


# In[ ]:


#test data
# read test dataset
testdataset = pd.read_csv('D:\BENSON\Rabala\test_data.csv)
#get only the temperature column
testdataset = testdataset.iloc[:30,3:4].values
real_temperature = pd.read_csv('data (12).csv')
real_temperature = real_temperature.iloc[30:,3:4].values
testing = sc.transform(testdataset)
testing = np.array(testing)
testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1)


# In[ ]:


#test model
predicted_temperature = regressor.predict(testing)
predicted_temperature = sc.inverse_transform(predicted_temperature)
predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))

