# model.fit의 반환값

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))

size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)
dataset = split_x(a, size)
x = dataset[:,:4]
y = dataset[:,4]

print(dataset.shape)    #(96, 5)
print(x.shape)          #(96, 4)
print(y.shape)          #(96,)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)

#2. 모델로드
model = load_model('../data/h5/save_keras35.h5')
model.add(Dense(5, name='king1'))
model.add(Dense(1, name='king2'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x,y,epochs = 1000, batch_size = 8, validation_split=0.2, callbacks=[early])

print(hist)
print(hist.history.keys())  #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
print(hist.history['loss'])

# 그래프
# val_loss와 loss의 간격이 좁을수록 좋은 성능
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()