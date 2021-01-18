#LSTM

#1. 데이터
import numpy as np
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
print (x.shape) #(4,3)
print (y.shape) #(4,)

x = x.reshape(4,3,1) #([[[1],[2],[3]], [[2], [3], [4]], [[4], [5], [6]]])

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape = (3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

'''
#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7])  #(3.)  LSTM에서 쓸 수 있는 데이터 구조 3차원 -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print(result)
'''