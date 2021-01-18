# 사이킷런
# LSTM으로 모델린
# Dense와 성능비교
# 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
dataset = load_breast_cancer()

print(dataset.DESCR)  
print(dataset.feature_names)  

x = dataset.data
y = dataset.target

# print(x.shape)  #(569, 30)
# print(y.shape)  #(569,)
# #--->데이터셋은 31 컬럼구성
# print(x[:5])
# print(y)    #0 , 1 두가지로 569개 출력

# 전처리 : minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =  train_test_split(x, y, train_size=0.8, random_state = 120)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, input_shape = (x.shape[1],1), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#hidden이 없는 모델 가능

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=2000, batch_size = 50, validation_data = (x_val, y_val), callbacks=[early_stopping])

loss, acc = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('mae : ', acc)

#이진분류 0,1 출력
y_pre = model.predict(x_test[:20])
y_pre = np.transpose(y_pre)
# print('y_pre : ', y_pre)
print('y값 : ', y_test[:20])

y_pre = np.where(y_pre<0.5, 0, 1)
# y_pre = np.argmax(y_pre, axis=1)
print(y_pre)


'''
Dense모델 : 
loss :  0.22489522397518158
acc :  0.9736841917037964
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]

LSTM모델 : 
loss :  0.3300505578517914
mae :  0.06349937617778778
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 0]]
'''