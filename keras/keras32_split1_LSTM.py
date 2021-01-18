#실습:  모델을 구성하시오

import numpy as np

#1. 데이터
a = np.array(range(1,11))
size = 5

''' def split_x(seq, size):
    x = []
    y = []
    for i in range(len(seq)-size+1):
        subset_x = seq[i:(i+size-1)]
        subset_y = seq[(i+size-1):(i+size)]
        x.append(subset_x)
        y.append((subset_y))
    return np.array(x), np.array(y)

x, y= split_x(a, size) '''

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)
dataset = split_x(a, size)
x = dataset[:,:4]
y = dataset[:,4]

x_pre = np.array([7,8,9,10])
x_pre = x_pre.reshape(1,-1)

print(x)
print(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 110)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_pre = x_pre.reshape(1,4,1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape = (x.shape[1], 1), activation='linear'))
model.add(Dense(30, activation='linear'))
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
4.591625474859029e-05
[[11.025031]]

'''