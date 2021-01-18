# 사이킷런
# LSTM으로 모델린
# Dense와 성능비교
# 회귀

#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# x 데이터 전처리 (MinMaxScaler) : 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train , y_train ,train_size=0.8, shuffle = True, random_state = 66)


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

print(x.shape)  #(506, 13, 1)
print(y.shape)  #(506,)

# y=y.reshape(506,1)


#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(x.shape[1],1))
dense = LSTM(100, activation='relu')(input1)
dense = Dense(10, activation='relu')(dense)
dense = Dense(20, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(80, activation='relu')(dense)
dense = Dense(384, activation='relu')(dense)
# dense = Dense(100, activation='relu')(dense)
# dense = Dense(120, activation='relu')(dense)
output = Dense(1)(dense)
model = Model(inputs=input1, outputs=output)


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#EarlyStopping정의 및 사용
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.fit(x_train,y_train, epochs=2000, batch_size=50, validation_data=(x_val, y_val), callbacks=[early_stopping]) 

#4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))

''' 
Dense모델 : 
loss :  7.246347427368164
mae :  2.1537396907806396
rmse :  2.6919046845498684
r2 :  0.9133034676451246


LSTM모델 sklearn : 
loss :  14.792984008789062
mae :  2.723668336868286
rmse :  3.8461649439028887
r2 :  0.8230142987559236

loss :  10.28790283203125
mae :  2.5314018726348877
rmse :  3.207476340630827
r2 :  0.8769138199239507

loss :  9.708100318908691
mae :  2.2550456523895264
rmse :  3.1157827692749662
r2 :  0.8838506688659079

loss :  9.50676155090332
mae :  2.2904999256134033
rmse :  3.083303042946316
r2 :  0.8862595888811198

loss :  8.797823905944824
mae :  2.25276255607605
rmse :  2.966112897719301
r2 :  0.8947413670437011
'''
 
