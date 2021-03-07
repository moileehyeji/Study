# <autoencoder : 입력과 출력이 같은 구조>
# 비지도학습 ex)PCA
# Bottleneck Hiddenlayer
# Bottleneck layer는 다음과 같이 표현하기도 합니다
#   Latent Variable
#   Feature
#   Hidden representation
# [수식]
#   Input Data를 Encoder Network에 통과시켜 압축된 z값을 얻습니다
#   압축된 z vector로부터 Input Data와 같은 크기의 출력 값을 생성합니다
#   이때 Loss값은 입력값 x와 Decoder를 통과한 y값의 차이입니다
# [학습 방법]
#   Decoder Network를 통과한 Output layer의 출력 값은 Input값의 크기와 같아야 합니다(같은 이미지를 복원한다고 생각하시면 될 것 같습니다)
#   이때 학습을 위해서는 출력 값과 입력값이 같아져야 합니다  
# https://deepinsight.tistory.com/126

# 2번카피
# 딥하게 구성
# 노드가 대칭을 이루도록 구성해야하는데 한것과 안한것 test
# 하나는 원칙전 오토인코더
# 결과 유사

import numpy as np 
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.
# print(x_train)
# print(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder1(): #대칭구조
    model = Sequential()
    model.add(Dense(units=16, input_shape = (784,), activation='relu'))  #units :  양의 정수, 출력 공간의 차원.
    model.add(Dense(units=32, activation='relu'))  
    model.add(Dense(units=64, activation='relu'))  
    model.add(Dense(units=32, activation='relu')) 
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(): #대칭아님
    model = Sequential()
    model.add(Dense(units=16, input_shape = (784,), activation='relu'))  #units :  양의 정수, 출력 공간의 차원.
    model.add(Dense(units=32, activation='relu'))  
    model.add(Dense(units=64, activation='relu'))  
    model.add(Dense(units=128, activation='relu')) 
    model.add(Dense(units=784, activation='sigmoid'))
    return model    

model_1 = autoencoder1()
model_2 = autoencoder2()

model_1.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model_2.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])


model_1.fit(x_train, x_train, epochs=10)
model_2.fit(x_train, x_train, epochs=10)


output_1 = model_1.predict(x_test)
output_2 = model_2.predict(x_test)

# 시각화
import matplotlib.pyplot as plt 
import random
fig, axes = plt.subplots(3, 5, figsize = (15,15))

random_imgs = random.sample(range(output_1.shape[0]), 5)
outputs = [x_test, output_1, output_2]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
            ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap = 'gray')
            ax.grid()
            ax.set_xticks([])
            ax.set_yticks([])

plt.show()