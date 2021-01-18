# keras18_boston4_MinMaxScaler2 소스 개선
# 실습 : validation 분리하여 전처리한뒤 모델 완성
# validation_split -> validation_data


#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
'''
print('=======================================================================================')
print('x.shape : \n',x.shape) # (506,13)
print('y.shape : \n',y.shape) # (506, )
# --> mlp모델
print('=======================================================================================')
print(x[:5])
print(y[:10])
# --> numpy는 부동소수점에 강한 단순연산에 강함
# --> 훈련을 잘 시키기 위해서는 데이터 정제(6가지 방법중 0~1사이 표시)가 필요
print('=======================================================================================')
print('x 최대값 최소값 : ',np.max(x), np.min(x))
# --> 해당 데이터는 교육용데이터
print('=======================================================================================')
print('컬럼명 : ',dataset.feature_names)
# print('컬럼묘사 : ',dataset.DESCR)
print('=======================================================================================')
'''
# 데이터 전처리 (MinMaxScaler) -> 한계: 컬럼별로 전처리X
# x = (x-np.min(x)) / (np.max(x)-np.min(x))
# x = x / 711. # 711. : 실수형 형변환

# #한계 1
# print(np.max(x[0])) # 396.9 -->CRIM열의 최대값 (전체를 711로 나눈것은 문제 O)

# X 데이터 전처리  (MinMaxScaler)2 ->데이터의 최대,최소값 알 필요X
from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler()
# scalar.fit(x)
# x = scalar.transform(x)

# #한계 2 : 각 컬럼별 최대 최소
# print('x 최대값 최소값 : ',np.max(x), np.min(x)) # 711.0 0.0 => 1.0, 0.0
# print(np.max(x[0])) # 0.9999999999


y=y.reshape(506,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)


# X_TRAIN 데이터 전처리 (MinMaxScaler)3 : 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# validation_data 실습
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)
x_val = scaler.transform(x_val)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense = Dense(10, activation='relu')(input1)
dense = Dense(30, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(123, activation='relu')(dense)
dense = Dense(384, activation='relu')(dense)
dense = Dense(100, activation='relu')(dense)
#dense = Dense(150, activation='relu')(dense)
dense = Dense(120, activation='relu')(dense)
output = Dense(1, activation='relu')(dense)
model = Model(inputs=input1, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=120, batch_size=20, validation_data = (x_val, y_val))

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

->통상적으로 x_train 전처리 후 성능이 더 좋아짐
'''
 
