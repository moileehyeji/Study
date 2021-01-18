# 로드모델 사용

import numpy as np

#1.
a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)

x = dataset[:,:4]
y = dataset[:,4]
x_pre = y[2:]
x_pre = x_pre.reshape(1,-1)

print(x)
print(y)
print(x_pre)
print(x.shape)
print(y.shape)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_pre = x_pre.reshape(1,4,1)

#2.
from tensorflow.keras.models import load_model

model = load_model('../data/h5/save_keras35.h5')

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=1, callbacks=[early])

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pre = model.predict(x_pre)

print(loss)
print(y_pre)

