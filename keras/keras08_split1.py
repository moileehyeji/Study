
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1.데이터
x=np.array(range(1,101)) #1~100
#x=np.array(range(100))  #0~99
y=np.array(range(101,201)) #101~200

#리스트의 슬라이싱
x_train=x[:60] #순서 0번째부터 59번째까지 ::: 값 1~60
x_val=x[60:80] #61~80
x_test=x[80:] #81~100

y_train=y[:60] #순서 0번째부터 59번째까지 ::: 값 1~60
y_val=y[60:80] #61~80
y_test=y[80:] #81~100

print(x_train)
print(x_val)
print(x_test)


#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(32))
model.add(Dense(134))
model.add(Dense(54))
model.add(Dense(87))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_data=(x_val,y_val))
#4.평가, 예측
results=model.evaluate(x_test,y_test,batch_size=1)
print('mse,mae: ',results)
y_predict= model.predict(x_test)
#print('y_predict: ',y_predict)


#사이킷런(sklearn)이란? 파이썬에서 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE: ',RMSE(y_test,y_predict))
#print('mae: ',mean_squared_error(y_test,y_predict))
print('mae: ',mean_squared_error(y_predict,y_test))


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('R2: ',r2)

