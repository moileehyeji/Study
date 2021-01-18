#실습 80 prdict

import numpy as np

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
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape = (3,1), activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(80,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(90,activation='relu'))
# model.add(Dense(35,activation='relu'))
# model.add(Dense(80,activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=30, mode = 'auto') 
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=2000, batch_size=7, callbacks=[early])

#4.평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_pre)
print(loss)
print(result)

""" 
전처리 전:
[0.0252032782882452, 0.13626839220523834]
[[80.49721]]

전처리 후:
[5.247901916503906, 1.7916405200958252]
[[84.77815]]

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=100, batch_size=1)

전처리 + 튜닝후 erlystopping527/1000, 레이어4와 노드 적게(LSTM layer노드는 크게):
[5.408135984907858e-05, 0.004778641741722822]
[[80.53015]]
[5.8228830312145874e-05, 0.00526057742536068]
[[80.4804]]

[2.5996316253440455e-05, 0.0037284446880221367]
[[80.393555]]

[2.4620549083920196e-05, 0.0032295079436153173]
[[80.054115]]
"""