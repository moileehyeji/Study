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

# hidden_layer_size
# unit :  양의 정수, 출력 공간의 차원.

import numpy as np 
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.
# print(x_train)
# print(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,), activation='relu'))  #unit :  양의 정수, 출력 공간의 차원.
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random 
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) =  plt.subplots(2, 5, figsize = (20,7))

#이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()