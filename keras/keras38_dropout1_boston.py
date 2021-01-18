# 드립아웃적용

#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

y=y.reshape(506,1)

# X_TRAIN 데이터 전처리 (MinMaxScaler)3 : 
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# validation_data 실습
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)
x_val = scaler.transform(x_val)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(13,))
dense = Dense(200, activation='relu')(input1)
danse = Dropout((0.3))(dense)
dense = Dense(30, activation='relu')(dense)
danse = Dropout((0.2))(dense)
dense = Dense(60, activation='relu')(dense)
danse = Dropout((0.2))(dense)
dense = Dense(123, activation='relu')(dense)
danse = Dropout((0.2))(dense)
dense = Dense(384, activation='relu')(dense)
danse = Dropout((0.3))(dense)
dense = Dense(100, activation='relu')(dense)
danse = Dropout((0.2))(dense)
dense = Dense(120, activation='relu')(dense)
danse = Dropout((0.2))(dense)
output = Dense(1, activation='relu')(dense)
model = Model(inputs=input1, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#EarlyStopping정의 및 사용
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=50, validation_data = (x_val, y_val), verbose=1, callbacks=[early_stopping]) # epochs= 54/2000

#4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ',mae)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))



''' 
전처리 전 (튜닝 후):
loss :  14.490835189819336
mae :  2.726719856262207
rmse :  3.8066831813943356
r2 :  0.8266292462780187

전처리 후(x/711.):
loss :  21.305908203125
mae :  3.689579725265503
rmse :  4.615832948370495
r2 :  0.7450925453518638

전처리 후(MinMaxScaler(x)):
loss :  15.450998306274414
mae :  2.4340524673461914
rmse :  3.9307764165416756
r2 :  0.8151416577344601

전처리 후(MinMaxScaler(x_train))(validation_split):
loss :  16.452713012695312
mae :  2.586681365966797
rmse :  4.056194459564113
r2 :  0.8031570315782633

전처리 후(MinMaxScaler(x_train))(validation_data):
loss :  9.41877269744873
mae :  2.296194553375244
rmse :  3.069002105194733
r2 :  0.8873122407232421

loss :  8.924368858337402
mae :  2.248032331466675
rmse :  2.9873691083292475
r2 :  0.8932273203803165

loss :  7.246347427368164
mae :  2.1537396907806396
rmse :  2.6919046845498684
r2 :  0.9133034676451246

->통상적으로 x_train 전처리 후 성능이 더 좋아짐

DROP OUT 후:
loss :  5.854621410369873
mae :  1.7973308563232422
rmse :  2.419632996019973
r2 :  0.9299543181619981
'''
 
