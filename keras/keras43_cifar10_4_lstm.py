from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

print(x_train.shape)    #(50000, 32, 32, 3)
print(x_test.shape)     #(10000, 32, 32, 3)
print(y_train.shape)    #(50000, 1)
print(y_test.shape)     #(10000, 1)

print(x_test[0].shape)  #(32, 32, 3)
print(y_test[0].shape)

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train.reshape(-1, 32*32, 3)/255.
x_test = x_test.reshape(-1, 32*32, 3)/255.
# x_val = x_val.reshape(-1,4*4, 147)/255.

#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape =(32*32, 3) ,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(152, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(95, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(54, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=90, callbacks=[early], validation_split=0.7)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

'''
cifar10_CNN : 
[1.6055035591125488, 0.6452999711036682]
y_pred[:10] :  [3 8 8 0 6 6 1 6 5 1]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]

cifar10_DNN :  
[1.7642892599105835, 0.36070001125335693]
y_pred[:10] :  [8 9 8 0 4 6 3 6 4 8]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]

cifar10_LSTM : 
[2.3040122985839844, 0.10000000149011612]
y_pred[:10] :  [3 3 3 3 3 3 3 3 3 3]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]
'''
