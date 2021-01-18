from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

x_train = x_train.reshape(-1,7*7, 16)/255.
x_test = x_test.reshape(-1,7*7, 16)/255.
x_val = x_val.reshape(-1,7*7, 16)/255.

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape =(7*7, 16) ,activation='relu'))
# model.add(Dense(152, activation='relu'))
# model.add(Dense(95, activation='relu'))
# model.add(Dense(54, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
path = '../data/modelcheckpoint/k45_fashion_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=7, batch_size=90, callbacks=[early, mc], validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

#시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('Lost Cost')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c='red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c='blue', label = 'val_acc')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.show()



'''
fashion_mnist_CNN : 
[0.924037516117096, 0.9107999801635742]
y_pred[:10] :  [9 2 1 1 0 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]

fashion_mnist_DNN :  
[1.1293410062789917, 0.8921999931335449]
y_pred[:10] :  [9 2 1 1 6 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]

fashion_mnist_LSTM : 
[1.081375002861023, 0.8568999767303467]
y_pred[:10] :  [9 2 1 1 0 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]
'''
