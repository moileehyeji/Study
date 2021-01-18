# 실습: validation_data를 만들것
# train_test_split사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터
x=np.array(range(1,101))
y=np.array(range(1,101))

'''
#리스트의 슬라이싱 1
x_train=x[:60] #순서 0번째부터 59번째까지 ::: 값 1~60
x_val=x[60:80] #61~80
x_test=x[80:] #81~100

y_train=y[:60] #순서 0번째부터 59번째까지 ::: 값 1~60
y_val=y[60:80] #61~80
y_test=y[80:] #81~100
'''


#리스트의 슬라이싱 2 : 훈련,테스트 데이터 무작위 슬라이싱(전체구간 훈련이 가능해짐), 순차적 슬라이싱
from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.6) #train데이터 무작위 60%
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True) #train데이터 순차적60%

#실습풀기
x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,test_size=0.2,shuffle=True)

print(x_train)
print(x_val)

#print(x_train) 

#갯수
print(x_train.shape) #64
print(x_val.shape) #16
print(y_train.shape)
print(y_val.shape)

#2.모델구성
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(120))
model.add(Dense(253))
model.add(Dense(187))
model.add(Dense(167))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,validation_data=(x_val,y_val)) #80 train 데이터중 16개 validation

#4.평가 예측
y_predict=model.predict(x_test)
print(y_predict)

loss, mae=model.evaluate(x_test, y_test)
print('loss: ', loss)
print('mae: ', mae)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE: ',RMSE(y_test,y_predict))

print('R2: ',r2_score(y_test,y_predict))

#성능차이
#shuffle=False
#loss:  0.0012189248809590936
#mae:  0.03451118618249893

#shuffle=True(성능 높아짐)
#loss:  0.0004431034903973341
#mae:  0.01736392453312874

#validation=0.2(성능 높아짐)
#loss:  0.00027271942235529423
#mae:  0.013236594386398792
#선생님은 성능 떨어짐-훈련양이 적어져서 예상

#실습결과
#loss:  0.0003190194838680327
#mae:  0.01472458802163601

#loss:  0.0002517066604923457
#mae:  0.014511192217469215
#RMSE:  0.015865266452051052
#R2:  0.9999998123748233

