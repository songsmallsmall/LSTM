# LSTM
import pandas as pd
from numpy import concatenate
import numpy as np
from math import sqrt
from pandas import datetime
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score


data = pd.read_excel('pmif.xlsx',header=0)
data['production_experience_activity_expectations'] = data['production_experience_activity_expectations'].fillna(data['production_experience_activity_expectations'].mean())
data['date'] = pd.to_datetime(data['date'])
data.set_index(['date'], inplace=True)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = data.values / 100
values = values.astype('float32')
reframed = series_to_supervised(values, 1, 1)

values = reframed.values
train = values[:40, :]
test = values[40:, :]
train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape)


model = Sequential()
model.add(Embedding(output_dim=50, input_dim=40, input_length=27))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dense(units=1, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


yhat = model.predict(test_X)
print('explained_variance_score:',explained_variance_score(test_y, yhat))
print('mean_absolute_error:',mean_absolute_error(test_y, yhat))
print('mean_squared_error:',mean_squared_error(test_y, yhat))
print('median_absolute_error:',median_absolute_error(test_y, yhat))
print('r2_score:',r2_score(test_y, yhat))

fig = plt.figure()
plt.xticks(rotation=45)
ax1 = fig.add_subplot(111)
ax1.plot(test_y, 'r', label="true", )
ax1.plot(yhat, 'g', label="predict", linestyle='--')
ax1.legend(loc=0)
plt.show()
