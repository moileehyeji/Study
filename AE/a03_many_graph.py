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
# 히든레이어의 노드가 클수록 이미지 변화가 적어진다.  
# pca와 유사

# 1. 데이터
import numpy as np 
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.
# print(x_train)
# print(x_test)


# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)


# 3. 컴파일, 훈련
print('node 1개 시작')
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train, x_train, epochs=10)

print('node 2개 시작')
model_02.compile(optimizer='adam', loss='binary_crossentropy')
model_02.fit(x_train, x_train, epochs=10)

print('node 4개 시작')
model_04.compile(optimizer='adam', loss='binary_crossentropy')
model_04.fit(x_train, x_train, epochs=10)

print('node 8개 시작')
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train, x_train, epochs=10)

print('node 16개 시작')
model_16.compile(optimizer='adam', loss='binary_crossentropy')
model_16.fit(x_train, x_train, epochs=10)

print('node 32개 시작')
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train, x_train, epochs=10)


# 4. 평가, 예측
output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)



# 시각화
import matplotlib.pyplot as plt 
import random
fig, axes = plt.subplots(7, 5, figsize = (15,15))

random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
            ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap = 'gray')
            ax.grid()
            ax.set_xticks([])
            ax.set_yticks([])

plt.show()
