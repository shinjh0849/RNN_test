from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



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
testset = pd.read_csv('TestsetTransformed.csv', header=0, index_col=0)

# making both train set & test set csv to sequential data
values_train = dataset.values
encoder_train = LabelEncoder()
values_train = values_train.astype('float32')
scaler_train = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler_train.fit_transform(values_train)
reframed_train = series_to_supervised(scaled_train, 1, 1)
reframed_train.drop(reframed_train.columns[[0]], axis=1, inplace=True)

values_test = testset.values
encoder_test = LabelEncoder()
values_test = values_test.astype('float32')
scaler_test = MinMaxScaler(feature_range=(0,1))
scaled_test = scaler_test.fit_transform(values_test)
reframed_test = series_to_supervised(scaled_test, 1, 1)


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

values_test = reframed_test.values
test = values_test[:, :]
test_X = test[:, :]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# using SMOTE to resolve class imbalance problem
sm = SMOTE(ratio='auto', kind='regular')
ros = RandomOverSampler(random_state=0)
resam_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# X_resampled, y_resampled = ros.fit_resample(resam_X, train_Y)
X_resampled, Y_resampled = sm.fit_resample(resam_X, train_Y)

X_resampled = X_resampled.reshape((X_resampled.shape[0], 1, X_resampled.shape[1]))
print(X_resampled.shape)

# design the model
model = Sequential()
model.add(LSTM(50, input_shape=(X_resampled.shape[1], X_resampled.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adagrad')

history = model.fit(X_resampled, Y_resampled, epochs=200, batch_size=72, 
					validation_data=(validation_X, validation_Y), verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valdation')
plt.legend()
plt.show()

# evaluating model
yhat_valid = model.predict(validation_X)
yhat_valid = np.round(yhat_valid)
DataFrame(yhat_valid).to_csv('validation.csv')

print('f1_score: ', f1_score(validation_Y, yhat_valid))
print('accuracy: ', accuracy_score(validation_Y, yhat_valid))

yhat_test = model.predict(test_X)
yhat_test = np.round(yhat_test)

DataFrame(yhat_test).to_csv('prediction.csv')

print('num of 0: ', (DataFrame(yhat_test)[[0]]==0).sum())
print('num of 1: ', (DataFrame(yhat_test)[[0]]==1).sum())


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