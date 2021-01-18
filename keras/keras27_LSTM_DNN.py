# keras23_LSTM3_scale을 DNN으로 코딩
# 결과치 비교
# 23번 파일보다 성능 좋게 만들것


import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pre = np.array([50,60,70])
x_pre = x_pre.reshape(1,3)

print(x.shape)  #(13,3)
print(y.shape)  #(13,)

#전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim =3 , activation='linear'))
model.add(Dense(30,activation='linear'))
model.add(Dense(90,activation='linear'))
model.add(Dense(80,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode = 'auto') 
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=1000, batch_size=5, callbacks=[early])

#4.평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_pre)
print(loss)
print(result)

'''
1. RNN 성능
[0.0252032782882452, 0.13626839220523834]
[[80.49721]]

2.DNN 성능
x 전처리 전:
[8.859270224093052e-07, 0.0005948726902715862]
[[79.996895]]
[0.0003690865996759385, 0.0141597343608737]
[[80.08179]]

x 전처리 후:
model = Sequential()
model.add(Dense(10, input_dim = 3, activation='linear'))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode = 'auto') 
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=100, batch_size=1)

[0.3032025694847107, 0.4348212480545044]
[[81.92996]]


RNN x전처리 튜닝 후:
[5.408135984907858e-05, 0.004778641741722822]
[[80.53015]]

DNN :
[4.735148906707764, 1.4031111001968384]
[[89.83262]]
DNN 튜닝 후 linear설정 earlystopping 727/1000:
[2.045051281096555e-11, 3.1728011435916414e-06]
[[80.00002]]
[3.1242015918753197e-11, 3.943076535506407e-06]
[[80.00002]]



'''