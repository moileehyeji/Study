#keras32_split1_LSTM Dense모델

#1. 데이터
import numpy as np

#시계열데이터 for문
a = np.array(range(1,11))
size = 5

def split_x (seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
dataset = split_x(a, size)
x = dataset[:,:4]
y = dataset[:,4]

x_pre = y[2:]
x_pre = x_pre.reshape(1,4)

print(x)
print(y)
print(x_pre)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim = 4, activation='relu'))
model.add(Dense(30, activation='relu'))
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
LSTM모델:
4.591625474859029e-05
[[11.025031]]

Dense 모델:
5.939390121056931e-06
[[11.023872]]
'''

