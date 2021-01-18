# 사이킷런
# LSTM으로 모델린
# Dense와 성능비교
# 다중

import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
# x, y = load_iris(return_X_y=True)
dataset = load_iris()
x = dataset.data
y = dataset.target

print(dataset.feature_names)    #(150,4)
print(dataset.DESCR)
print(x.shape)  #(150, 4)
print(y.shape)  #(150,)
print(x[:5])
print(y)        
""" 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2] 
 #꽃이 50개씩 3종류 --> shuffle 필수
 """

""" # y 전처리(Keras) : train_test_split 전후 상관없음
# 원핫인코딩(One-Hot Encoding)
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical

y = to_categorical(y)
print(y)
print(y.shape)  #(150, 3) """

# y 전처리(sklearn)
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()

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

#3차원
x = x.reshape(x.shape[0], x.shape[1],1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, input_shape = (x.shape[1],1), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(3, activation='softmax')) 

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k45_iris_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early = EarlyStopping(monitor = 'loss', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=2000, batch_size=50 ,validation_data=(x_val, y_val), callbacks=[early, mc])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss, mae :', loss)

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
loss, acc : [0.054814912378787994, 1.0]

LSTM모델 sklearn : 
loss, acc : [0.007020933087915182, 1.0]
'''