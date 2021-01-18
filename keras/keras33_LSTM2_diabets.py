# 사이킷런
# LSTM으로 모델린
# Dense와 성능비교
# 회귀

import numpy as np

#1. 데이터
from sklearn.datasets import load_diabetes

dataset =load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 110)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 110)

#데이터 전처리3
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
from tensorflow.keras.layers import Dense, Input, LSTM


input = Input(shape=(x.shape[1],1))
dense = LSTM(100, activation='linear')(input)
dense = Dense(50, activation='linear')(dense)
dense = Dense(50, activation='linear')(dense)
dense = Dense(50, activation='linear')(dense)
dense = Dense(50, activation='linear')(dense)
output = Dense(1)(dense)
model = Model(inputs=input, outputs=output) 


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=2000, batch_size=50, validation_data = (x_val, y_val), callbacks=[early_stopping])

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
'''