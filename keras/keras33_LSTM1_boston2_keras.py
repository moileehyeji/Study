# 케라스
# LSTM으로 모델린
# Dense와 성능비교
# # 회귀

# 사이킷런
# LSTM으로 모델린
# Dense와 성능비교
# 회귀

#1.데이터
import numpy as np
#샘플데이터 로드
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 108)

#데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#3차원
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

print(x_train.shape)
print(x_test.shape)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(100, input_shape = (x_train.shape[1],1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(451, activation='relu'))
model.add(Dense(234, activation='relu'))
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
Dense모델 : 
loss :  9.41877269744873
mae :  2.296194553375244
rmse :  3.069002105194733
r2 :  0.8873122407232421

LSTM모델 sklearn : 
loss :  20.309154510498047
mae :  3.5534865856170654
rmse :  4.506568745594623
r2 :  0.7570178494906212

LSTM모델 keras : 
loss, mae :  24.344141006469727 2.855922222137451
RMSE :  4.933978390570761
R2 :  0.70755625560934
'''