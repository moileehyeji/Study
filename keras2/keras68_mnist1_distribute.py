# gpu 여러개 동시에 돌리기

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

# X 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

# 다중분류
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ModelCheckpoint : earlystopping되기전 최적의 가중치 저장
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}_{val_loss:.4f}.hdf5'
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

#------------------------------------------------------------------- 다중gpu 분산처리
# 선택적으로 돌릴수도 있다!
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
        tf.distribute.HierarchicalCopyAllReduce())
#-------------------------------------------------------------------


with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=200, kernel_size=(2,2), padding='same', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=200, kernel_size=2, padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=100, kernel_size=2, padding='same', strides=4))
    model.add(Flatten())
    model.add(Dense(520, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 3. 컴파일, 훈련

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
#################################################

hist = model.fit(x_train, y_train, epochs=7, batch_size=20, callbacks=[early], validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss : ', loss[0])
print('acc : ', loss[1])




'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]
'''







