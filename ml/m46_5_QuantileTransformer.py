# keras18_boston6_StandardScaler 카피

# 이상치제어에 효과적인 전처리2222222222222222222222

#----------------------------------------
# aaa = np.array([[1,2,-1000,3,4,6,7,8,90,100,5000],          
# ---> standard scalar 전처리하면 데이터가 모여있지않고 흩어질 것
# 그렇다면 중위값을 1로 scaling하자
# m46_4_robust.py

# RobustScaler
# : 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
#   IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.

# QuantileTransformer
# : 1000개의 분위수로 고정
# 분위수 정보를 사용하여 특성을 변환
# 주어진 특성에 대해이 변환은 가장 빈번한 값을 분산시키는 경향 
# 또한 (한계) 이상치의 영향을 줄임
# 따라서 이것은 강력한 전처리 체계
# 이상치 제어에 효과적


#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# =================================데이터 전처리 (QuantileTransformer) 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# scalar = QuantileTransformer()   # output_distribution='uniform' --> 균등분포
scalar = QuantileTransformer(output_distribution='normal')# 변환 된 데이터의 한계 분포. -->정규분포

scalar.fit(x)
x = scalar.transform(x)


y=y.reshape(506,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)

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
# dense = Dense(150, activation='relu')(dense)
dense = Dense(120, activation='relu')(dense)
output = Dense(1, activation='relu')(dense)
model = Model(inputs=input1, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_split=0.4, verbose=2)

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

->통상적으로 전처리 후 성능이 더 좋아짐

전처리 후(StandardScaler(x))
loss :  16.968822479248047
mae :  2.56310772895813
rmse :  4.119322807091346
r2 :  0.7969822438560329

robustscalar() 전처리 후:
loss :  20.226205825805664
mae :  2.898548126220703
rmse :  4.497355432951508
r2 :  0.7580103483321317

QuantileTransformer() 전처리후:
loss :  14.698819160461426
mae :  2.8061022758483887
rmse :  3.8339038550305022
r2 :  0.8241409163299861
'''
 
