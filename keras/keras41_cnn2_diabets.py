# CNN으로 구성
# 2차원을 4차원으로 늘려서 하시오.


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape)  #(442, 10)
print(y.shape)  #(442,)

#전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#4차원
x_train = x_train.reshape(-1, 10, 1, 1)
x_test = x_test.reshape(-1, 10, 1, 1)
x_val = x_val.reshape(-1, 10, 1, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 500,kernel_size=(2,1),input_shape = (10,1,1)))
model.add(MaxPooling2D(pool_size=1))
# model.add(Conv2D(filters = 256,kernel_size=1))
# model.add(Conv2D(filters = 125,kernel_size=1))
# model.add(Conv2D(filters = 100,kernel_size=1))
# model.add(Conv2D(filters = 80,kernel_size=1))
model.add(Flatten())
# model.add(Dense(200,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode= 'auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=10, callbacks=[early])

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

print('loss : ', loss)
print('mae : ',mae)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))

x_pre = x_test[:10]
y_pre = model.predict(x_pre)
print('y_pred[:10] : ', y_pre.reshape(1,-1))
print('y_test[:10] : ', y_test[:10])


'''
DNN모델 : 
loss :  2166.6650390625 
mae :   38.769779205322266
RMSE :  46.54745007951129
R2 :    0.6297812819678937

LSTM모델 : 
loss :  2578.437255859375 
mae :   40.43071746826172
rmse :  50.778316251131514
r2 :    0.5594216266685046

CNN모델 : 
loss :  3345.784423828125
mae :  46.672889709472656
rmse :  57.842756300367
r2 :  0.48447401699219805
y_pred[:10] :  [[150.90688  194.207    167.21431  105.20254  109.582855 116.17901
  114.81565   84.326164 194.82518   90.07186 ]]
y_test[:10] :  [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
'''