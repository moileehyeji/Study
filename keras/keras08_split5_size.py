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


#리스트의 슬라이싱 2
from sklearn.model_selection import train_test_split

# total size=0.9
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False) #shuffle=False일 경우 x_train:1~70, x_test=71~90/shuffle=True일경우 1~100 무작위
# total size=1.1
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2 , shuffle=False) #->에러

print(x_train)
print(x_test)

#갯수
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



'''
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
model.fit(x_train,y_train,epochs=100) 

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

'''