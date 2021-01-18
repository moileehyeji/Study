# keras42_mnist4_fashion -> Conv1D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1.데이터
#샘플데이터 로드
x_train = np.load('../data/npy/fashion_x_train.npy')
x_test = np.load('../data/npy/fashion_x_test.npy')
y_train = np.load('../data/npy/fashion_y_train.npy')
y_test = np.load('../data/npy/fashion_y_test.npy')

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train.reshape(-1,7*7, 16)/255.
x_test = x_test.reshape(-1,7*7, 16)/255.
x_val = x_val.reshape(-1,7*7, 16)/255.

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters = 100,strides=2,kernel_size=2, padding='same',input_shape =(7*7, 16) ,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
path = '../data/modelcheckpoint/k54_8_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=90, callbacks=[early, mc], validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss, acc : ', loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

'''
fashion_mnist_CNN : 
[0.924037516117096, 0.9107999801635742]

fashion_mnist_DNN :  
loss, acc :[1.1293410062789917, 0.8921999931335449]

fashion_mnist_LSTM : 
loss, acc :[1.081375002861023, 0.8568999767303467]

fashion_mnist_Conv1D
loss, acc :[0.9277617335319519, 0.8687999844551086]
'''
