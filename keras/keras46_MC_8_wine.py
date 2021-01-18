#다중분류

import numpy as np

#데이터 로드
from sklearn.datasets import load_wine
dataset = load_wine()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)    #0,1,2(다중분류)
print(x.shape)  #(178, 13)
print(y.shape)  #(178, )

# y 전처리(Keras) : train_test_split 전후 상관없음
# 원핫인코딩(One-Hot Encoding)
from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
print(y)
print(y.shape)  #(178, 3)


# 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

# x 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_shape = (13,), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k45_wine_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early = EarlyStopping(monitor = 'acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_data=(x_val, y_val), 
                callbacks=[early, mc])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss, acc :', loss)

y_pre = model.predict(x_test[:10])
# print('y_pre : \n', y_pre)
print('y_pre2 : \n', y_pre)
print('y실제값 \n: ', y_test[:10])


#결과치 나오게 코딩할 것 : argmax
y_pre = np.argmax(y_pre, axis=1)
print('y_pre : \n', y_pre)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

plt.title('Loss Cost')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

plt.show()

'''
Dense모델 : 
loss, acc : [0.030613545328378677, 1.0]
'''

