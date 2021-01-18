# keras23_LSTM3_scale -> Conv1D

import numpy as np
import pandas as pd

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pre = np.array([50,60,70])
x_pre = x_pre.reshape(1,3)

print(x.shape)  #(13,3)
print(y.shape)  #(13,)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

x = x.reshape(13,3,1)
x_pre = x_pre.reshape(1,3,1)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters = 100,kernel_size=1 ,input_shape = (3,1)))
model.add(Conv1D(filters = 80,kernel_size=2 ))
model.add(Conv1D(filters = 50,kernel_size=1 ))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(1))

model.summary()
'''
#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=30, mode = 'auto') 
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=2000, batch_size=7, callbacks=[early])

#4.평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_pre)
print('loss, mae : ', loss)
print(result)
'''
""" 
LSTM : 
loss, mae :[2.4620549083920196e-05, 0.0032295079436153173]
[[80.054115]]

Conv1D : 
loss, mae : [3.559934611985227e-06, 0.0012718713842332363]
[[79.998474]]
"""