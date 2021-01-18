# 주말과제
# dense 모델로 구성 input_shape = (28*28, )

#1. 데이터
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(x_test.shape)     #(10000, 28, 28)
print(y_train.shape)    #(60000,)
print(y_test.shape)     #(10000,)

#전처리
x_train = x_train.reshape(x_train.shape[0], 28*28)/255.
x_test = x_test.reshape(x_test.shape[0], 28*28)/255.

print(x_train.shape)   
print(x_test.shape)    

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_shape=(784,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode= 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=500, batch_size=200, callbacks=[early])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)


x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
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
'''
