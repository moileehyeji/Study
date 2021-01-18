from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

x_train = np.load('../data/npy/fashion_x_train.npy')
x_test = np.load('../data/npy/fashion_x_test.npy')
y_train = np.load('../data/npy/fashion_y_train.npy')
y_test = np.load('../data/npy/fashion_y_test.npy')

#전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle = True, random_state=66)

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

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)/255.

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 50, kernel_size=2, input_shape =(28,28,1), strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters = 50, kernel_size=3, strides=1, padding='valid'))
model.add(MaxPooling2D(pool_size=3))
model.add(Flatten())
model.add(Dense(232, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
early = EarlyStopping(monitor='acc', patience=20, mode='auto')
path = '../data/modelcheckpoint/k50_fashion_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=30,validation_data=(x_val, y_val) ,callbacks=[early, mc])

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
[1.5017063617706299, 0.9056000113487244]
y_pred[:10] :  [9 2 1 1 0 1 4 6 5 7]
y_test[:10] :  [[9 2 1 1 6 1 4 6 5 7]]

'''