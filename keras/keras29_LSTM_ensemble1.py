# LSTM 앙상블 모델을 만드시오
# predict는 85의 근사치

import numpy as np

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50],[40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120],
              [2,3,4], [3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,0,11,12,13,50,60,70]) 
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

#전처리
x1_predict = x1_predict.reshape(1,-1)
x2_predict = x2_predict.reshape(1,-1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1)
scaler.fit(x2)
x1 = scaler.transform(x1)
x2 = scaler.transform(x2)
x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)

#3차원 변경
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM

input1 = Input(shape = (x1.shape[1], 1))
lstm1 = LSTM(300, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(lstm1)
dense1 = Dense(20, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)


input2 = Input(shape=(x2.shape[1], 1))
lstm2 = LSTM(300, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(lstm2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)


#모델병합
merge = concatenate([dense1, dense2])
dense = Dense(200, activation='relu')(merge)
dense = Dense(200, activation='relu')(dense)
dense = Dense(400, activation='relu')(dense)
dense = Dense(100, activation='relu')(dense)

output = Dense(1)(dense)

model = Model(inputs = [input1, input2], outputs = output)

# model.summary()

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=30, mode = 'auto') 
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit([x1, x2],y,epochs=2000, batch_size=7, callbacks=[early])

#4.평가, 예측
loss = model.evaluate([x1,x2],y)
result = model.predict([x1_predict, x2_predict])
print(loss)
print(result) 

'''
[0.0031253970228135586, 0.041049279272556305]
[[60.681694]]

[2.421463966369629, 1.141822099685669]
[[85.61063]]
'''