
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
# ModelCheckpoint : earlystopping되기전 최적의 가중치 저장
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}_{val_loss:.4f}.hdf5'
early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=7, batch_size=20, callbacks=[early, cp], validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=90)
print('loss : ', loss[0])
print('acc : ', loss[1])

# 시각화
#한글폰트 사용 실습===========================================================
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
#===========================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))      

plt.subplot(2,1,1)  #2행 1열중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('손실비용')
# plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)  #2행 2열중 두번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('정확도')
# plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['accuracy','val_accuracy'])

plt.show()


""" plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red',label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', c='red',label = 'val_loss') """

'''
x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
print('y_test[:10] : ', y_test_pre)

print(x_test[10].shape)
'''


'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]
'''







