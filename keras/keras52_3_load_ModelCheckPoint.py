# modelcheckpoint 사용
# --> 모델(model.save), 가중치(model.save_weights) 각각 model 생성한 결과 loss 값 동일
# --> modelcheckpoint 사용시 통상적으로 성능 높아짐
# 성능 평가지표 최우선 : loss

import numpy as np
import matplotlib.pyplot as plt

# 1. mnist 데이터 셋
x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)   ->흑백(60000, 28, 28, 1)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])

# print(x_train[0].shape) #(28, 28)

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

# model1 = Sequential()
# model1.add(Conv2D(filters=200, kernel_size=(2,2), padding='same', input_shape = (28,28,1)))
# model1.add(MaxPooling2D(pool_size=2))
# model1.add(Dropout(0.2))
# model1.add(Conv2D(filters=200, kernel_size=2, padding='same', strides=2))
# model1.add(MaxPooling2D(pool_size=2))
# model1.add(Conv2D(filters=100, kernel_size=2, padding='same', strides=4))
# model1.add(Flatten())
# model1.add(Dense(520, activation='relu'))
# model1.add(Dense(200, activation='relu'))
# model1.add(Dense(15, activation='relu'))
# model1.add(Dense(10, activation='softmax'))

# 모델 save1================================================================
# model.save('../data/h5/k52_1_model1.h5')

# 3. 컴파일, 훈련
# ModelCheckpoint : earlystopping되기전 최적의 가중치 저장
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/k52_mnist_{epoch:02d}_{val_loss:.4f}.hdf5'
# k52_1_mnist_??? => k52_1_MCK.hdf5 이름으로 바꿀 것
# early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# hist = model.fit(x_train, y_train, epochs=7, batch_size=20, callbacks=[early, cp], validation_split=0.2)

# # 모델 save2=======모델, 가중치, 훈련==========================================
# model.save('../data/h5/k52_1_model2.h5')        
# # ================가중치, 훈련=================================================
# model.save_weights('../data/h5/k52_1_weight.h5') 


# # 모델1 load_weights============================================================
# model1.load_weights('../data/h5/k52_1_weight.h5')


# # 4-1. 평가, 예측
# loss1 = model1.evaluate(x_test, y_test, batch_size=90)
# print('가중치_loss : ', loss1[0])
# print('가중치_acc : ', loss1[1])


# # 모델2 load_model===============================================================
# # 상단 모델구성과는 별도
# model2 = load_model('../data/h5/k52_1_model2.h5')   
# # 4-2. 평가, 예측
# loss2 = model2.evaluate(x_test, y_test, batch_size=90)
# print('로드모델_loss : ', loss2[0])
# print('로드모델_acc : ', loss2[1])


# 모델3 modelcheckpoint===============================================================
model3 = load_model('../data/modelcheckpoint/k52_1_checkpoint.hdf5')   
# 4-2. 평가, 예측
loss3 = model3.evaluate(x_test, y_test, batch_size=90)
print('로드체크가중치_loss : ', loss3[0])
print('로드체크가중치_acc : ', loss3[1])



'''
#####   모델, 가중치 save모델의 결과 비교 : 동일

model 1 :
가중치_loss :  0.24505098164081573
가중치_acc :  0.9276000261306763

model 2 :
로드모델_loss :  0.24505098164081573
로드모델_acc :  0.9276000261306763

model 3 :  성능 가장 높음
로드체크가중치_loss :  0.23103635013103485
로드체크가중치_acc :  0.9319000244140625

'''







