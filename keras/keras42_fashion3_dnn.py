from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(x_test.shape)     #(10000, 28, 28)
print(y_train.shape)    #(60000,)
print(y_test.shape)     #(10000,)

print(x_train[0])
print(y_train)   

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

""" 
ValueError: Found array with dim 3. MinMaxScaler expected <= 2.
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val) """

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train.reshape(-1,(28*28))/255.
x_test = x_test.reshape(-1,(28*28))/255.
x_val = x_val.reshape(-1,(28*28))/255.

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape =((28*28), ) ,activation='relu'))
# model.add(Dense(152, activation='relu'))
# model.add(Dense(95, activation='relu'))
# model.add(Dense(54, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=90, callbacks=[early], validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

'''
fashion_mnist_CNN : 
[0.924037516117096, 0.9107999801635742]
y_pred[:10] :  [9 2 1 1 0 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]

fashion_mnist_DNN :  
[1.1293410062789917, 0.8921999931335449]
y_pred[:10] :  [9 2 1 1 6 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]
'''
