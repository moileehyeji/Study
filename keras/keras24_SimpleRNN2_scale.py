#실습 keras23_LSTM3_scale 복사
# SimpleRNN vs RSTM

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, input_shape = (3,1), activation='linear'))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=100, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_pre)
print(loss)
print(result)

""" 
1. LSTM
[0.20295092463493347, 0.285023033618927]
[[79.34567]]
[0.0252032782882452, 0.13626839220523834]
[[80.49721]]
[0.9567981958389282, 0.6573229432106018]
[[83.23564]]
[0.09924127161502838, 0.22790725529193878]
[[79.91056]]
[0.03603596240282059, 0.1582343727350235]
[[80.60374]]

2. SimpleRNN
[0.5419584512710571, 0.4983307421207428]
[[82.92777]]
[0.00449881749227643, 0.04294389858841896]
[[79.94371]]
[0.0026959495153278112, 0.032495059072971344]
[[80.04177]]
[8.96629280759953e-05, 0.008668734692037106]
[[79.97992]]
[0.058488938957452774, 0.19582384824752808]
[[80.78912]]
"""