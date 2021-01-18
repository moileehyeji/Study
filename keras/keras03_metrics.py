import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8]) 
y_test = np.array([6,7,8])

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(295))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(1))

#3. 모델컴파일, 훈련
#metrics: 평가지표 
#accuracy=acc : 전체 샘플중 맞게 예측한 샘플수(정확한 값:1.0)
#mse:평균제곱오차
#mae: 절대평균오차
#model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #0.9999=!1,2.=!2 근접값이라도 형태및 값이 다르면 accuracy이 1.0이 안나옴
#model.compile(loss='mse',optimizer='adam',metrics=['mse']) #loss값과 동일
model.compile(loss='mse',optimizer='adam',metrics=['mae']) 
model.fit(x_train,y_train,epochs=100,batch_size=1) 

#4.모델평가, 예측
loss=model.evaluate(x_test,y_test,batch_size=1) 
print('loss: ',loss)

#result=model.predict([9])
result=model.predict(x_train)
print('result: ', result)
