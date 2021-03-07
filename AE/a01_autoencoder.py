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


#마지막레이어 : decoded = Dense(784, activation='sigmoid')
#컴파일:  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')  
#         autoencoder.compile(optimizer='adam', loss='mse')

import numpy as np 
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.
# print(x_train)
# print(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)      # 히든레이어의 노드가 클수록 이미지 변화가 적어진다.  
decoded = Dense(784, activation='sigmoid')(encoded)    # activation='relu'일때 하단의 연산부분이 날라가면서 이미지의 잡음이 많이 생김
#hidden layer가 1개인 단순 모델

autoencoder = Model(input_img, decoded)

autoencoder.summary()



# loss='binary_crossentropy' & loss='mse' 결과동일
# acc 안좋음 
# WHY? (784, activation='sigmoid') ---> loss로 평가
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  
autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])

autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, validation_split=0.2)

decoded_img = autoencoder.predict(x_test)

# encode, decode 이미지 보기
import matplotlib.pyplot as plt
n=10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# 이미지의 불필요하게 튀어나온 부분이 없어짐