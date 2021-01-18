from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

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
model.add(LSTM(100, input_shape =(32*32, 3) ,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(152, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(95, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(54, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=13, batch_size=90, callbacks=[early], validation_split=0.7)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))


'''
cifar100_CNN :
[3.795083999633789, 0.3871999979019165]
y_pred[:10] :  [68 65 37 72 71  6 44 86 23 10]
y_test[:10] :  [[49 33 72 51 71 92 15 14 23  0]] 

cifar100_DNN :  
[4.288858413696289, 0.04399999976158142]
y_pred[:10] :  [69 66 21 66 53 53 66 66 69 66]
y_test[:10] :  [[49 33 72 51 71 92 15 14 23  0]]

cifar100_LSTM : 
[4.611823081970215, 0.009999999776482582]
y_pred[:10] :  [35 35 35 35 35 35 35 35 35 35]
y_test[:10] :  [[49 33 72 51 71 92 15 14 23  0]]

'''
