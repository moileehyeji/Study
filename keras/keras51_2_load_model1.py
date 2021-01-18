# 모델 생성
# 1. 모델구성 아래 저장한 모델  -->모델저장

import numpy as np
import matplotlib.pyplot as plt

# 1. mnist 데이터 셋
x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')

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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

''' model = Sequential()
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
model.add(Dense(10, activation='softmax')) '''


# 모델 load1 :  모델 저장==================================================================
model = load_model('../data/h5/k51_1_model1.h5')
model.summary()
#=========================================================================================


# 3. 컴파일, 훈련
# ModelCheckpoint : earlystopping되기전 최적의 가중치 저장
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k51_mnist_{epoch:02d}_{val_loss:.4f}.hdf5'
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=7, batch_size=20, callbacks=[early, cp], validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
mnist_CNN : 
loss :  0.2662297487258911
acc :  0.9259999990463257
'''







