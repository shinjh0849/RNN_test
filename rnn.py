from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# start of main
trainset_file = './Trainset.csv'
testset_file = './Testset.csv'
output_file = './Testset_Prediction.csv'

COLUMNS = ['time', 'temp', 'max_peak', 
           'sig_peak', 'avg_peak', 'freq', 'label']
FEATURES = ['time', 'temp', 'max_peak', 
            'sig_peak', 'avg_peak', 'freq']
LABEL = [0, 1]

# reading the initial file
train_set = pd.read_csv(trainset_file, engine='python', skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv(testset_file, engine='python', skipinitialspace=True, skiprows=1, names=FEATURES)


# replacing missing values to 0
train_set.fillna(0, inplace=True)
test_set.fillna(0, inplace=True)
test_set.to_csv('TestsetTransformed.csv')
train_set.to_csv('TrainsetTransformed.csv')

# saving the transformed file
dataset = pd.read_csv('TrainsetTransformed.csv', header=0, index_col=0)
testset = pd.read_csv('TestsetTransformed.csv', engine='python', skipinitialspace=True, skiprows=1, names=FEATURES)

values_train = dataset.values
encoder = LabelEncoder()
values_train = values_train.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(values_train)
reframed_train = series_to_supervised(scaled_train, 1, 1)

# init
# values_test = testset.values
# values_test = values_test.astype('float32')
# scaled_test = scaler.fit_transform(values_test)
# reframed_test = series_to_supervised(scaled_test, 1, 1)

# # defining the model
# values_train = reframed_train.values
# values_test = reframed_test.values
# train = values_train[:, :]
# test = values_test[:, :]
# train_X, train_Y = train[:, :-1], train[:, -1]
# test_X, test_Y = test[:, :-1], test[:, -1]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)



# second try
values = reframed_train.values
train = values[:, :]
validation = values[:, :]
# split into input and outputs
train_X, train_Y = train[:, :-1], train[:, -1]
validation_X, validation_Y = validation[:, :-1], validation[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))
print(train_X.shape, train_Y.shape, validation_X.shape, validation_Y.shape)



# design the model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adagrad')

history = model.fit(train_X, train_Y, epochs=500, batch_size=72, validation_data=(validation_X, validation_Y), verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valdation')
plt.legend()
plt.show()

# evaluating model
#yhat = model.predict(test_X)

# plotting the dataset
# groups = [0, 1, 2, 3, 4, 5, 6]
# i = 1
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# plt.show()

# convert series to supervised learning
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

