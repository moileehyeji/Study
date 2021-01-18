# 네이밍 룰(암묵적)
# keras00_핵심용어

import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
#패키지를 하위까지 정의하지 않으면 기능 사용시 상위패키지 직접 기재
#from tensorflow.keras import models
#from tensorflow import keras
from tensorflow.keras.layers import Dense


#1. 데이터 [1,2,3,4,5,5,6,7,8]
#훈련하는 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
#훈련하지 않는 데이터
x_test = np.array([6,7,8]) 
y_test = np.array([6,7,8])

#2. 모델구성
model=Sequential()
#패키지를 정의하지 않으면 상위패키지 직접 기재
#model=models.Sequential()
#model=keras.models.Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(295))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(1))

#3. 모델컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1) #배치사이즈는 데이터갯수보다 숫자가 더 커도 내부적으로 조절이 됨

#4.모델평가, 예측
loss=model.evaluate(x_test,y_test,batch_size=1) #compile 'mse'와 연결
print('loss: ',loss)

result=model.predict([9])
result2=model.predict([10])
result3=model.predict(x_test)
print('result: ', result)
print('result2: ', result2)
print('result3: ', result3)

#실습 epochs=100으로 result 9의 근접값 도출(하이퍼 파라미터 튜닝)
#Dense의 node값이 너무 크면 ERROR
#layer와 node수 증감으로 도출
