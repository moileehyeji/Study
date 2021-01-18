# CNN으로 구성
# 2차원을 4차원으로 늘려서 하시오.


import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x, y = load_wine(return_X_y=True)
print(x.shape)  #(178, 13)
print(y.shape)  #(178,)

#전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#4차원
x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)
x_val = x_val.reshape(-1, 13, 1, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 500,kernel_size=(2,1),input_shape = (13,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 256,kernel_size=1))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 125,kernel_size=1))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 100,kernel_size=1))
model.add(Conv2D(filters = 80,kernel_size=1))
model.add(Flatten())
# model.add(Dense(200,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=10, callbacks=[early])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

print('loss : ', loss)
print('acc : ',acc)
y_predict = model.predict(x_test)

x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre.reshape(1,-1))
print('y_test[:10] : ', y_test_pre)


'''
DNN모델 : 
loss : 0.030613545328378677
acc : 1.0

LSTM모델 : 
loss : 0.39956337213516235
acc : 0.8888888955116272

CNN모델 : 
0.18509119749069214
loss :  0.18509119749069214
acc :  0.9722222089767456
y_pred[:10] :  [[2 1 1 0 1 1 2 0 0 2]]
y_test[:10] :  [2 1 1 0 1 1 2 0 0 1]
'''