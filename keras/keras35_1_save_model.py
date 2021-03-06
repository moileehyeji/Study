# 모델 저장 및 재사용

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(200, input_shape = (4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 모델 저장
model.save("../data/h5/save_keras35.h5")       #저장성공
# model.save("..//data//h5//save_keras35_1.h5")   #저장성공
# model.save("..\data\h5\save_keras35_2.h5")     #저장성공   ->\n와 같은 예약어는 error
# model.save("..\\data\\h5\\save_keras35_3.h5")   #저장성공
