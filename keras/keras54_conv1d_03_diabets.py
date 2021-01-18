# keras33_LSTM1_diabets -> Conv1D

#1.데이터
import numpy as np

#샘플데이터 로드
x = np.load('../data/npy/diabets_x.npy')
y = np.load('../data/npy/diabets_y.npy')

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 110)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 110)

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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

input = Input(shape=(x.shape[1],1))
dense = Conv1D(filters = 300,kernel_size=2, strides=1, padding='valid' ,activation='relu')(input)
dense = MaxPooling1D(pool_size=2)(dense)
dense = Dropout(0.3)(dense)
dense = Conv1D(filters = 20,kernel_size=2, strides=1, padding='valid' ,activation='relu')(dense)
dense = Flatten()(dense)
dense = Dense(80, activation='relu')(dense)
dense = Dense(90, activation='relu')(dense)
dense = Dense(210, activation='relu')(dense)
dense = Dropout(0.2)(dense)
dense = Dense(150, activation='relu')(dense)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)
model = Model(inputs=input, outputs=output) 


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k54_3_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=2000, batch_size=50, validation_data = (x_val, y_val), callbacks=[early_stopping, mc])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))

'''
Dense모델 : 
loss, mae :  2166.6650390625 38.769779205322266
RMSE :  46.54745007951129
R2 :  0.6297812819678937

LSTM모델 : 
loss, mae :  2578.437255859375 40.43071746826172
rmse :  50.778316251131514
r2 :  0.5594216266685046

Conv1D
loss, mae :  2085.394287109375 36.618099212646484
rmse :  45.66612357134248
r2 :  0.6436679568820876
'''