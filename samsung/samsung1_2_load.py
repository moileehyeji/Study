
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 load
x = np.load('./samsung/npy/samsung_x.npy')
y = np.load('./samsung/npy/samsung_y.npy')

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True, random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.2, shuffle = True, random_state=110)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2.
model = load_model('./samsung/h5/sam3_model_89431.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=20)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))

x_pred1 = np.array([[90300., 91400., 87800., -5885518., -4498684.]])
x_pred2 = np.array([[89800., 91200., 89100., -1781416., -2190214.]])
x_pred1 = scaler.transform(x_pred1)
x_pred2 = scaler.transform(x_pred2)
x_pred1 = x_pred1.reshape(x_pred1.shape[0], x_pred1.shape[1], 1)
x_pred2 = x_pred2.reshape(x_pred2.shape[0], x_pred2.shape[1], 1)
print('x_pred(89,700) : ' , model.predict(x_pred1))
print('x_pred(89,200) : ' , model.predict(x_pred2))


