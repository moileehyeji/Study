import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=1, activation='linear')) #하이퍼 파라미터 튜닝
model.add(Dense(10,activation='linear'))
model.add(Dense(73))
model.add(Dense(41))
model.add(Dense(100))
model.add(Dense(102))
model.add(Dense(10))
model.add(Dense(1)) 

model.summary()

""" 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=1, activation='linear')) #하이퍼 파라미터 튜닝
model.add(Dense(3,activation='linear'))
model.add(Dense(4))
model.add(Dense(1)) 

model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________ 
"""

# 실습2 + 과제
# ensemble1,2,3,4에 대해 서머리 계산하고 
# 이해한 것을 과제로 제출할 것
# layer를 만들 때 'name'이란 것에 대해 확인하고 설명할 것(왜 하는지)
# name을 반드시 써야할 때가 있다. 그때를 말해라
