# keras33_LSTM1_cancer -> Conv1D

#1.데이터
import numpy as np

#샘플데이터 로드
x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')

# 전처리 : minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =  train_test_split(x, y, train_size=0.8, random_state = 120)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

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

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters = 100, kernel_size=2 ,input_shape = (x.shape[1],1),
                strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 50, kernel_size=2 , strides=2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#hidden이 없는 모델 가능

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k54_4_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=2000, batch_size = 50, validation_data = (x_val, y_val), callbacks=[early_stopping, mc])

loss, acc = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('mae : ', acc)

#이진분류 0,1 출력
y_pre = model.predict(x_test[:20])
y_pre = np.transpose(y_pre)
# print('y_pre : ', y_pre)
print('y값 : ', y_test[:20])

y_pre = np.where(y_pre<0.5, 0, 1)
# y_pre = np.argmax(y_pre, axis=1)
print(y_pre)


'''
Dense모델 : 
loss :  0.22489522397518158
acc :  0.9736841917037964
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]

LSTM모델 : 
loss :  0.3300505578517914
mae :  0.06349937617778778
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 0]]

Conv1D모델:
loss :  0.16524851322174072
mae :  0.05259440839290619
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]
'''