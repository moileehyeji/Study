# keras40_mnist4_lstm -> Conv1D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
#1.데이터

#샘플데이터 로드
x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

x_train = x_train.reshape(-1, (28*28) , 1)/255.
x_test = x_test.reshape(-1, (28*28) , 1)/255.
x_val = x_val.reshape(-1, (28*28) , 1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters = 100,kernel_size=2 ,input_shape = ((28*28) , 1), strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 70,kernel_size=2 , strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters = 50,kernel_size=2 , strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3.컴파일, 훈련
path = '../data/modelcheckpoint/k54_7_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, batch_size=90, epochs=1000, validation_data=(x_val, y_val), callbacks=[early, mc])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss, acc : ',loss)

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

mnist_Conv1D :
loss, acc :  [0.2294730693101883, 0.9742000102996826]
y_pred[:10] :  [7 2 1 0 4 1 4 9 6 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]
'''


