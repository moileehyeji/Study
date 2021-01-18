
# 실습 
# 지표는 acc (0.985 이상)

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (???)
# y_pred[:10] = (???)


import numpy as np
import matplotlib.pyplot as plt

# 1. mnist 데이터 셋
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)   ->흑백(60000, 28, 28, 1)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

print(x_train[0].shape) #(28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# X 전처리
# 이미지의 특성을 찾아 숫자를 맞춰야 함 3차원--> 4차원 --> float타입 변경 -->/255 (0~1 수렴) ~~~~~~>전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# 다중분류
# Y 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=500, kernel_size=(2,2), padding='same', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=200, kernel_size=2, padding='same', strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=100, kernel_size=2, padding='same', strides=4))
model.add(Flatten())
model.add(Dense(520, activation='relu'))
model.add(Dense(200, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=90, callbacks=[early])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print(loss)

x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
print('y_test[:10] : ', y_test_pre)

print(x_test[10].shape)
""" import matplotlib.pyplot as plt

plt.imshow(x_test[10], 'gray')
plt.show() """

'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]
'''







