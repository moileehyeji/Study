# gpu 여러개 각각 돌리기


import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------gpu 여러개 각각 돌리기
import tensorflow as tf
gpus = tf.config.experimenter.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimenter.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)
#-------------------------------------------------------------------



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


#-----------------------------------------------------------------BatchNormalization
# kernel_initializer='he_normal'
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2
model = Sequential()
model.add(Conv2D(filters=200, kernel_size=(2,2), padding='same', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=200, kernel_size=2, padding='same', strides=2, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=100, kernel_size=2, padding='same', strides=4, kernel_regularizer=l1(0.01)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(520, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=7, batch_size=20, callbacks=[early], validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss : ', loss[0])
print('acc : ', loss[1])


# [0.15593186020851135, 0.9835000038146973]
# ->
# loss :  0.3966255187988281
# acc :  0.9635000228881836




