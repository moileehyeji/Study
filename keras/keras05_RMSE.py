
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1.데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=array([12,13,14,15])
y_test=array([12,13,14,15])
x_pred=array([16,17,18])

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(67))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_split=0.2)#0.2만큼 검증

#4.평가, 예측
results=model.evaluate(x_test,y_test,batch_size=1)
print('mse,mae: ',results) #loss와 metrics에 있는 값 출력

y_predict= model.predict(x_test)
#print('y_predict: ',y_predict)


#사이킷런(sklearn)이란? 파이썬에서 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리
from sklearn.metrics import mean_squared_error

#해당 라이브러리엔 RMSE가 없음 그러므로 함수를 만들어 값 루트씌워 반환
#sqrt:루트 ===>np.sqrt(result[0])
#파이썬은 줄바꿈 탭으로 범위 표시
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE: ',RMSE(y_test,y_predict))
#print('mae: ',mean_squared_error(y_test,y_predict))
print('mae: ',mean_squared_error(y_predict,y_test))


#RMSE를 0.1로 낮춰라 -> 하이퍼파라미터 튜닝