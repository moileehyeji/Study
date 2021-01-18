# keras23_LSTM3_scale 소스를 카피해서
# LSTM층을 두개 만들어라
# --> LSTM이 1층 이상이면 성능이 떨어짐

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

# x = x.reshape(13,3,1)
# x_pre = x_pre.reshape(1,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)
x_pre = x_pre.reshape(x_pre.shape[0], x_pre.shape[1], 1)


#2.모델구성
# return_sequences=True : LSTM층의 2차원출력 특성을 3차원 출력으로 바꿔줌
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (3,1), activation='relu', return_sequences=True))  #(None, 3, 1) 을 받음
model.add(LSTM(10, activation='relu'))  #(None, 3, 10) 을 받음
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(35,activation='relu'))
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
print(loss)
print(result)
'''
'''
LSTM 1layer:
[5.565991159528494e-05, 0.005902546923607588]
[[80.044136]]

LSTM 2layer:
[0.0014249677769839764, 0.030273113399744034]
[[[79.9574 ]
  [78.35668]
  [79.56343]]]

LSTM 3layer:
[5.166819095611572, 1.6322836875915527]
[[[79.33444 ]
  [77.74049 ]
  [79.690956]]]
'''