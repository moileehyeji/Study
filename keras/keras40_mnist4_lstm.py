# 주말과제
# lstm 모델로 구성 input_shape = (28*28, 1)
# lstm 모델로 구성 input_shape = (28*14, 2)
# lstm 모델로 구성 input_shape = (28*7, 4)
# lstm 모델로 구성 input_shape = (7*7, 16)

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#전처리
# x_train = x_train.reshape(-1, (28*28) , 1)/255.
# x_test = x_test.reshape(-1, (28*28) , 1)/255.
# x_train = x_train.reshape(-1, (28*14) , 2)/255.
# x_test = x_test.reshape(-1, (28*14) , 2)/255.
# x_train = x_train.reshape(-1, (28*7) , 4)/255.
# x_test = x_test.reshape(-1, (28*7) , 4)/255.
# x_train = x_train.reshape(-1, (7*7) , 16)/255.
# x_test = x_test.reshape(-1, (7*7) , 16)/255.
x_train = x_train.reshape(-1, 1 , 16*49)/255.
x_test = x_test.reshape(-1, 1 , 16*49)/255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape = (1, 16*49), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3.컴파일, 훈련
early = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, batch_size=90, epochs=1000, callbacks=[early])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)

x_pre = x_test[:10]
y_pred = model.predict(x_pre)
y_pred = np.argmax(y_pred, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1) 
print('y_pred[:10] : ',  y_pred)
print('y_test[:10] : ', y_test_pre)

'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_DNN : 
[0.28995245695114136, 0.9696999788284302]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_LSTM :
[0.1378639042377472, 0.9803000092506409]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

'''


