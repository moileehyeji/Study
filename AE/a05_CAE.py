# 4번 카피
# CNN으로 딥하게 구성
# 2개의 모델을 만드는데 하나는 원칙적 오토인코더
# 다른 하나는 랜덤하게 만들고 싶은대로 히든구성
# 2개의 성능 비교

# 이미지에 적용되는 autoencoder는 항상 convolutional autoencoder입니다. 왜냐하면 성능이 더 좋기 때문이죠.
# 인코더는 Conv2D와 MaxPooling2D layer의 층으로 구성되고, 디코더는 Conv2D와 UpSampling2D layer의 층으로 구성
# fit할 때 x_train_out 따로 만들어
# 별차이 없음

import numpy as np 
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1)/255.
# print(x_train)
# print(x_test)

x_train_out = x_train.reshape(60000,784).astype('float32')/255

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, UpSampling2D

def autoencoder1(): #대칭구조
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=4, input_shape = (28,28,1), padding='same', activation='relu'))  
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu')) 
    model.add(UpSampling2D(2))
    model.add(Flatten())
    model.add(Dense(units=8))
    model.add(Dense(units=16))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(): #대칭아님
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=4, input_shape = (28,28,1), padding='same', activation='relu'))  
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')) 
    model.add(UpSampling2D(2))
    model.add(Flatten())
    model.add(Dense(units=64))
    model.add(Dense(units=16))
    model.add(Dense(units=784, activation='sigmoid'))
    return model    

model_1 = autoencoder1()
model_2 = autoencoder2()

model_1.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model_2.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])


model_1.fit(x_train, x_train_out, epochs=10)
model_2.fit(x_train, x_train_out, epochs=10)


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