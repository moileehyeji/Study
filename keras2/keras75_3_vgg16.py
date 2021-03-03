# 남이 만든 모델 내가 활용하는 법
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',       #imagenet데이터로 저장된 가중치
            include_top=False,          #False일때 input_shape 변경가능 
            input_shape = (32,32,3))    
# print(vgg16.weights)  # layer 13개


vgg16.trainable = False # 동결(freezen)
vgg16.summary()
print(len(vgg16.weights))           #26  
print(len(vgg16.trainable_weights)) #0 

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #, activation='softmax'))

model.summary()
print('그냥 가중치의 수 : ', len(model.weights))           #26 -> 32
print('동결하기 전 훈련되는 가중치의 수 : ', len(model.trainable_weights)) #0  -> 6
""" 
Total params: 14,719,879
Trainable params: 5,191     #vgg16가중치만 빼서 쓰고 이 만큼만 훈련
Non-trainable params: 14,714,688
"""

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
'''
                                                                Layer Type Layer Name   Layer      Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000016A066EF550>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x0000016A0671BA60>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x0000016A06743880>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x0000016A3B129220>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x0000016A040DF1F0>             dense_2    True
'''



