# keras23_LSTM3_scale을 함수형으로 코딩

import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pre = np.array([50,60,70])

print(x.shape)  #(13,3)
print(y.shape)  #(13,)

x = x.reshape(13,3,1)
x_pre = x_pre.reshape(1,3,1)

#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape = (3,1))
lstm = LSTM(10, activation='linear')(input)
dense = Dense(70)(lstm)
dense = Dense(150)(dense)
dense = Dense(50)(dense)
output = Dense(1)(dense)

model = Model(inputs = input, outputs = output)


""" model = Sequential()
model.add(LSTM(10, input_shape = (3,1), activation='linear'))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) """

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=100, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_pre)
print(loss)
print(result)

'''
[37.812355041503906, 4.813446521759033]
[[65.633255]]
'''