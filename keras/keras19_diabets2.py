# 실습: 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개 파일 완성

#데이터전처리(계산)

import numpy as np

#1. 데이터
from sklearn.datasets import load_diabetes
dataset =load_diabetes()

x = dataset.data
y = dataset.target

print(x[:2])
print(y[:5])
print(x.shape) #(442,10)
print(y.shape) #(442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

#데이터 전처리1
x = (x-np.min(x)) / (np.max(x)-np.min(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 110)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

y=y.reshape(442,1)
input = Input(shape=(10,))
dense = Dense(10)(input)
dense = Dense(100)(dense)
dense = Dense(384)(dense)
dense = Dense(200)(dense)
dense = Dense(10)(dense)
output = Dense(1)(dense)
model = Model(inputs=input, outputs=output) 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))



'''
diabets1 결과:
loss, mae :  2324.1611328125 40.17805862426758
RMSE :  48.20955902871521
R2 :  0.6028698123414886

diabets2 결과:
loss, mae :  2347.20703125 40.44464874267578
RMSE :  48.447981617293834
R2 :  0.5989320479939803
'''
