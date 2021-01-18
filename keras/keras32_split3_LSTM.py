# 과제 및 실습 LSTM
# 전처리, EARLYSTOPPING, 등등
# 데이터 1~100 / 5개씩
#       x      /  y
# 1 2 3 4 5    /  6
# 95 96 97 98 99 / 100

# predict만들 것
# 96 97 98 99 100 -->101
# .... 
# 100 101 102 103 104 --> 105
# 예상 predict는 (101,102,103,104,105)

import numpy as np

#1. 데이터
data = np.array(range(1,101))
data_pre = np.array(range(96,106))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)
dataset = split_x(data, size)
dataset_pre = split_x(data_pre,size)
x = dataset[:,:5]
y = dataset[:,5]
x_pre = dataset_pre[:,:5]


print(x)
print(y)
print(x_pre)
print(x_pre.shape)
print(x.shape)
print(y.shape)
'''
#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_pre = x_pre.reshape(x_pre.shape[0], x_pre.shape[1],1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape = (x.shape[1], 1), activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=8, callbacks=[early])

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pre = model.predict(x_pre)

print(loss)
print(y_pre)
'''
'''
LSTM:
0.0005837779026478529
[[101.00243 ]
 [102.00638 ]
 [103.01052 ]
 [104.014854]
 [105.01942 ]]

'''