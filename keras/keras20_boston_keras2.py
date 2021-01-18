# 2개 파일을 만드시오.
# 1. EarlyStopping 을 적용하지 않은 최고의 모델
# 2. EarlyStopping 을 적용한 최고의 모델

import numpy as np
from tensorflow.keras.datasets import boston_housing

#1. 데이터
# sklearn과 x,y 나누는 방식이 다름
(x_train,y_train),(x_test, y_test) = boston_housing.load_data()

print('x_train.shape : ',x_train.shape) #(404,13)
print('x_test.shape : ',x_test.shape)   #(102,13)
print('y_train.shape : ',y_train.shape) #(404,)
print('y_test.shape : ',y_test.shape)   #(102,)

print(np.max(x_train), np.min(x_test)) #711.0, 0.0

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 108)

#데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(80, input_dim = 13))
model.add(Dense(100, activation='relu'))
# model.add(Dense(234, activation='relu'))
# model.add(Dense(384, activation='relu'))
model.add(Dense(451, activation='relu'))
# model.add(Dense(688, activation='relu'))
# model.add(Dense(456, activation='relu'))
model.add(Dense(234, activation='relu'))
# model.add(Dense(170, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=40, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data = (x_val, y_val), callbacks=[early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))


'''
1. x_train 전처리, validation_data
loss, mae :  15.290338516235352 2.7666664123535156
RMSE :  3.9102860267984063
R2 :  0.8163187180403131

1.EarlyStopping
loss, mae :  11.678218841552734 2.2771663665771484
RMSE :  3.41734102941071
R2 :  0.8597107183419552
'''