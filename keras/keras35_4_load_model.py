# 로드모델 사용 및 레이어추가

import numpy as np

#1.
a = np.array(range(1,11))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:,:5]
y = dataset[:,5]
x_pre = np.array(y)
x_pre = x_pre.reshape(1,5)

print(x, y)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_pre = x_pre.reshape(1,5,1)

#2.
#[ERROR] : Dense 레이어 이름 충돌
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

model = load_model('../data/h5/save_keras35.h5')   #name: dense_1 ~ 이름생성
model.add(Dense(5, name='kingkeras1'))          #name: dense_1 -> kingkeras1
model.add(Dense(1, name='kingkeras2'))          #name: dense_2 -> kingkeras2

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

