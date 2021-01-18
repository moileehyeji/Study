#샘플데이터 boston housing price

#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

'''
print(x.shape) # (506,13)
print(y.shape) # (506, )
# --> mlp모델
print('=======================================================================')
print(x[:5])
print(y[:10])
# --> numpy는 부동소수점에 강한 단순연산에 강함
# --> 훈련을 잘 시키기 위해서는 데이터 정제(6가지 방법중 0~1사이 표시)가 필요
print('=======================================================================')
print('x 최대값 최소값 : ',np.max(x), np.min(x))
# --> 해당 데이터는 교육용데이터
print('컬럼명 : ',dataset.feature_names)
print('컬럼묘사 : ',dataset.DESCR)
print('=======================================================================')

'''

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
dense = Dense(150, activation='relu')(dense)
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
loss :  14.490835189819336
mae :  2.726719856262207
rmse :  3.8066831813943356
r2 :  0.8266292462780187
 '''
