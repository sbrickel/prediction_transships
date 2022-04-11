# Prediction model transship for cargo type from historic data
# Copyright (C) 2022 Sebastian Brickel
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the 
# FreeSoftware Foundation; either version 2 of the License, or (at your 
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
# more details.
#
# You should have received a copy of the GNU General Public License along 
# with this program; if not, write to the Free Software Foundation, Inc., 59 
# Temple Place, Suite 330, Boston, MA 02111-1307 USA

# Requirements: 
# pip install openpyxl (for reading provided xlsx-file)
# pip install pandas numby (for doing maths)
# pip install keras tensorflow sklearn (for prediction model)

# Imports
import pandas as pd
import numpy as np

# We want to plot our results in the end
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

# File I/O
# Read in data file
df = pd.read_excel('VesselData.xlsx')

# Check read in was correct
#df.head
# ADD: data cleaning

# Extraxt subset of interest
df = df[['discharge1','load1','discharge2','load2','discharge3','load3',
    'discharge4','load4','vesselid']]

# Check data types 
df.dtypes

# Convert from integer to float (we want to do maths afterwards)
# not the most elegant way, but it works, a loop over each column would be nicer here
df["discharge1"] = pd.to_numeric(df["discharge1"], downcast="float")
df["discharge2"] = pd.to_numeric(df["discharge2"], downcast="float")
df["discharge3"] = pd.to_numeric(df["discharge3"], downcast="float")
df["discharge4"] = pd.to_numeric(df["discharge4"], downcast="float")
df["load1"] = pd.to_numeric(df["load1"], downcast="float")
df["load2"] = pd.to_numeric(df["load2"], downcast="float")
df["load3"] = pd.to_numeric(df["load3"], downcast="float")
df["load4"] = pd.to_numeric(df["load4"], downcast="float")
df["vesselid"] = pd.to_numeric(df["vesselid"], downcast="float")

# Data preparation
# Just for 1 in the beginning, can be extended to the four cargo types once it works
# Transships is defined as difference between discharge and load
data = pd.DataFrame(index=range(0,len(df)),columns=['transship','vesselid'])
for i in range(0,len(data)):
     data["transship"][i]=df['discharge1'][i] - df['load1'][i]
     data["vesselid"][i]=df['vesselid'][i]

# Min-max scaler - used for normalising the data, this helps the model to learn
scaler=MinMaxScaler(feature_range=(0,1))

data.index=data.vesselid
data.drop("vesselid",axis=1,inplace=True)

# Split around 80:20 for train:test - from person experience this performs well
train_size=int(len(df)*0.8)
valid_size=int(len(df)*0.2)

final_data = data.values
train_data=final_data[0:train_size,:]
valid_data=final_data[valid_size:,:]

scaler=MinMaxScaler(feature_range=(0,1))

# Split data into windows of size x, around 40 percent should give a good results
# according to literature
scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(2050,len(train_data)):
    x_train_data.append(scaled_data[i-2050:i,0])
    y_train_data.append(scaled_data[i,0])


# Long Short-Term Memory (LSTM) model should work based on this time-series datasets.
# Number layers has been set to 150 rather arbitrarily

# Unfortunately I can not test this, since there is an error in my Tensorflow installation
lstm_model=Sequential()
lstm_model.add(LSTM(units=150,return_sequences=True,input_shape=
    (np.shape(x_train_data)[1],1)))
lstm_model.add(LSTM(units=150))
lstm_model.add(Dense(1))

model_data=data[len(data)-len(valid_data)-2050:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)

# Prepare train and test data
# The model parameter were chosen out of recommendations from a quick internet search
# Several epochs should be tested to improve performence
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

X_test=[]
for i in range(2050,model_data.shape[0]):
    X_test.append(model_data[i-2050:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

# Prediction Function
predicted_transships=lstm_model.predict(X_test)
predicted_transships=scaler.inverse_transform(predicted_stock_price)

# Check accuracy of model, split around 80:20 for train:test
valid_data['Predictions']=predicted_transships
plt.plot(train_data["Transships"])
plt.plot(valid_data[['Transships',"Predictions"]])
