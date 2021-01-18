# keras33_LSTM1_iris -> Conv1D

#1.데이터
import numpy as np

#샘플데이터 로드
x = np.load('../data/npy/iris_x.npy')
y = np.load('../data/npy/iris_y.npy')

# y 전처리(sklearn)
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()

# 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

# x 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters = 100,kernel_size=2 ,input_shape = (x.shape[1],1), 
                    strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 50,kernel_size=2 ,strides=2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(3, activation='softmax')) 

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k54_5_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early = EarlyStopping(monitor = 'loss', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=2000, batch_size=50 ,validation_data=(x_val, y_val), callbacks=[early, mc])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss, acc :', loss)

y_pre = model.predict(x_test[:10])
# print('y_pre : \n', y_pre)
print('y_pre2 : \n', y_pre)
print('y실제값 \n: ', y_test[:10])


#결과치 나오게 코딩할 것 : argmax
y_pre = np.argmax(y_pre, axis=1)
print('y_pre : \n', y_pre)

'''
Dense모델 : 
loss, acc : [0.054814912378787994, 1.0]

LSTM모델 sklearn : 
loss, acc : [0.007020933087915182, 1.0]

Conv2D : 
loss, acc : [0.008645012974739075, 1.0]

'''