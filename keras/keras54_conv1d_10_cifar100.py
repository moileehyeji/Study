# keras44_mnist4_cifar100 -> Conv1D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1.데이터
#샘플데이터 로드
x_train = np.load('../data/npy/cifar100_x_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')

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
model.add(Dense(152, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(95, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(54, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
path = '../data/modelcheckpoint/k54_10_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=200, callbacks=[early, mc], validation_data = (x_val, y_val))

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

cifar100_Conv1D : 
[4.123187065124512, 0.16210000216960907]
y_pred[:10] :  [72 80 15 43 10 31 27 78 71 57]
y_test[:10] :  [[49 33 72 51 71 92 15 14 23  0]]

'''

