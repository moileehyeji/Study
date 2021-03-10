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

# MaxAbsScaler
# 최대절대값과 0이 각각 1, 0이 되도록 스케일링
# 절대값이 0~1사이에 매핑되도록 한다. 
# 즉 -1~1 사이로 재조정한다. 
# 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 
# 큰 이상치에 민감할 수 있다.

# PowerTransformer
# 데이터를 더 가우스와 비슷하게 만들려면 특징적으로 전력 변환을 적용
# Box-Cox는 입력 데이터가 엄격하게 양수 여야하는 반면 
# Yeo-Johnson은 양수 또는 음수 데이터를 모두 지원합니다.
# log화

#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# =================================데이터 전처리 (MaxAbsScaler) , (PowerTransformer)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
scalar = MaxAbsScaler()
# scalar = PowerTransformer(method='yeo-johnson')
# scalar = PowerTransformer(method='box-cox')   -> MinMaxScaler선행
# scaler = MinMaxScaler(feature_range=(1, 2))
# power = PowerTransformer(method='box-cox')
# pipeline = Pipeline(steps=[('s', scaler),('p', power)])
# x= pipeline.fit_transform(x)

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

MaxAbsScaler():
loss :  17.288455963134766
mae :  2.714048385620117
rmse :  4.157938239616017
r2 :  0.7931581378166699

PowerTransformer(method='yeo-johnson'):
loss :  598.9620361328125
mae :  22.701963424682617
rmse :  24.473699368593905
r2 :  -6.166079344718901
'''
 
