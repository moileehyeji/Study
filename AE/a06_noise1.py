import numpy as np 
from tensorflow.keras.datasets import mnist

# 95% PCA수치 154 가장 안정적인 수치로 이미지 복원이 되는지 확인하자


(x_train, _), (x_test, _) =mnist.load_data()   # y_train, y_test 빈자리

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.


# 노이즈 만들기
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)   #0~0.1값을 랜덤하게 더해서 노이즈 만들기 (0~1.1분포)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)  # (0~1.1분포) --> 1보다 크면 1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
# 0~1  --> 0~1 같은 범위이지만 노이즈가 생김

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)
# 95% PCA수치 154 가장 안정적인 수치로 이미지 복원이 되는지 확인하자

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10, batch_size=256)

output = model.predict(x_test_noised)


import matplotlib.pyplot as plt 
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5,figsize = (20,7))

#이미지 다섯개를 무작위로 고른다.
radom_imgs = random.sample(range(output.shape[0]), 5) 

#원본(입력) 이미지를 맨위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[radom_imgs[i]].reshape(28,28), cmap = 'gray')
    if i==0:
        ax.set_ylabel('INPUT', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([])  

#잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[radom_imgs[i]].reshape(28,28), cmap = 'gray')
    if i==0:
        ax.set_ylabel('NOISE', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([])   

#출력 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[radom_imgs[i]].reshape(28,28), cmap = 'gray')
    if i==0:
        ax.set_ylabel('OUTPUT', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([]) 

plt.tight_layout()
plt.show()
    
    