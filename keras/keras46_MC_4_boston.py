# CNN으로 구성
# 2차원을 4차원으로 늘려서 하시오.

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x, y = load_boston(return_X_y=True)
print(x.shape)  #(506, 13)
print(y.shape)  #(506,)

#전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#4차원
x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)
x_val = x_val.reshape(-1, 13, 1, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 200,kernel_size=(2,1),input_shape = (13,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(filters = 100,kernel_size=1))
model.add(Conv2D(filters = 80,kernel_size=1))
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k45_boston_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early = EarlyStopping(monitor='loss', patience=20, mode= 'auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2,validation_data=(x_val, y_val) ,callbacks=[early, mc])

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

print('loss : ', loss)
print('mae : ',mae)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))

x_pre = x_test[:10]
y_pre = model.predict(x_pre)
print('y_pred[:10] : ', y_pre.reshape(1,-1))
print('y_test[:10] : ', y_test[:10])

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

plt.subplot(1,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

plt.title('Loss Cost')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
# plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
# plt.grid()

# plt.title('Accuracy')
# plt.ylabel('acc')
# plt.xlabel('epochs')
# plt.legend(loc = 'upper right')

plt.show()

''' 
DNN모델 : 
loss :  7.246347427368164
mae :  2.1537396907806396
rmse :  2.6919046845498684
r2 :  0.9133034676451246


LSTM모델  : 
loss :  8.797823905944824
mae :  2.25276255607605
rmse :  2.966112897719301
r2 :  0.8947413670437011

CNN모델  : 
loss :  4.494823455810547
mae :  1.6760056018829346
rmse :  2.120099468937249
r2 :  0.9462232137123261
'''
 

