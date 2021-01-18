# keras43_mnist4_cifar10 -> Conv1D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1.데이터
#샘플데이터 로드
x_train = np.load('../data/npy/cifar10_x_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train.reshape(-1, 32*32, 3)/255.
x_test = x_test.reshape(-1, 32*32, 3)/255.
x_val = x_val.reshape(-1, 32*32, 3)/255.

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters = 10,kernel_size=2, strides=1, padding='same', input_shape =(32*32, 3) ,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(152, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(95, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(54, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
path = '../data/modelcheckpoint/k54_9_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=90, callbacks=[early,mc], validation_data = (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss, acc :', loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

'''
cifar10_CNN : 
loss, acc :[1.6055035591125488, 0.6452999711036682]

cifar10_DNN :  
loss, acc :[1.7642892599105835, 0.36070001125335693]

cifar10_LSTM : 
loss, acc :[2.3040122985839844, 0.10000000149011612]

cifar10_Conv1D : 
loss, acc :[1.941870093345642, 0.520799994468689]
'''
