from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy as np

x_train = np.load('../data/npy/cifar10_x_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.7, shuffle = True, random_state=66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 500, kernel_size=2, input_shape =(32, 32, 3), strides=2, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 300, kernel_size=3, strides=2, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 200, kernel_size=3, strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(283, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
path = '../data/modelcheckpoint/k50_cifar10_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[early, mc], validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

y_pred = model.predict(x_test[:10])
y_pred = np.argmax(y_pred, axis=1)
y_test_pred = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pred)
print('y_test[:10] : ', y_test_pred.reshape(1,-1))

'''
CNN:
[1.3478608131408691, 0.6403999924659729]
y_pred[:10] :  [3 8 8 0 6 6 1 6 2 1]
y_test[:10] :  [[3 8 8 0 6 6 1 6 3 1]]
'''