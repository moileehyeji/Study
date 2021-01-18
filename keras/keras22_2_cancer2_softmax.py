# keras21_cancer1를 다중분류로 코딩하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
dataset = load_breast_cancer()

print(dataset.DESCR)  
print(dataset.feature_names)  
x = dataset.data
y = dataset.target

print(x.shape)  #(569, 30)
print(y.shape)  #(569,)

""" # y 전처리1
from tensorflow.keras.utils import to_categorical
y = to_categorical(y) """

# y 전처리2
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()

# 전처리 : minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =  train_test_split(x, y, train_size=0.8, random_state = 104, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_shape = (30,), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data = (x_val, y_val), callbacks=[early])

loss, acc = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('acc : ', acc)

import tensorflow as tf
y_pre = model.predict(x_test[20:30])
y_pre = tf.argmax(y_pre, axis = 1)
print('y_pre : ', y_pre)
print('y값 : ', y_test[20:30])

""" 
<출력>
loss :  0.20605576038360596
acc :  0.9561403393745422
y_pre :  tf.Tensor([1 0 1 1 1 0 0 0 1 1], shape=(10,), dtype=int64)
y값 :  [[0. 1.]
 [1. 0.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.] 
 """