# 다차원 댄스모델
# (n, 32, 32, 3) -> (n, 32, 32, 3)

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)    #(50000, 32, 32, 3)
print(x_test.shape)     #(10000, 32, 32, 3)
print(y_train.shape)    #(50000, 1)
print(y_test.shape)     #(10000, 1)

print(x_test[0].shape)  #(32, 32, 3)
print(y_test[0].shape)

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

y_train = x_train
y_test = x_test 

#2. 모델구성
model = Sequential()
model.add(Dense(500, input_shape =(32, 32, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(283, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
# model.add(Flatten())
model.add(Dense(3))

# model.summary()     #(None, 32, 32, 3)


#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=2, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=256, callbacks=[early], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=256)
print(loss)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)     #(10000, 32, 32, 3)



'''
cifar10_CNN : 
[1.1453654766082764, 0.5907999873161316]
y_pred[:10] :  [3 8 8 8 6 6 1 2 5 1]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]

[2.1068367958068848, 0.6182000041007996]
y_pred[:10] :  [4 8 8 0 6 6 1 6 2 1]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]

[1.6055035591125488, 0.6452999711036682]
y_pred[:10] :  [3 8 8 0 6 6 1 6 5 1]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]
'''
