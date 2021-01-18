# 실습: 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개 파일 완성

#데이터전처리(MinMaxScaler) x_train데이터, validation_data

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
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_data = (x_val, y_val))

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

diabets3 결과:
loss, mae :  2417.564453125 40.72371292114258
RMSE :  49.16873780931149
R2 :  0.5869099801793163

diabets4 결과:
loss, mae :  2301.443603515625 39.61891174316406
RMSE :  47.97336898498221
R2 :  0.6067515498429323

diabets5 결과:
loss, mae :  2182.22119140625 39.15340805053711
RMSE :  46.71425083453319
R2 :  0.627123202706926

'''


