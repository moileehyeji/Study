import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns

# 데이터 로드
df = pd.read_csv('./samsung/csv/samsung_1+2.csv', index_col=0, header=0, encoding='cp949', thousands=',')
data = np.load('./samsung/npy/samsung0114_1.npy')

print(data)

x = data[:-1,[0,1,2,3,9,12]]         # x, y분리
y = data[1:,[3]]
y = y.reshape(-1,)

print(x.shape)  #(2397, 13)
print(y.shape)  #(2397,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True, random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.2, shuffle = True, random_state=110)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2.
model = Sequential()
model.add(Conv1D(filters = 230, kernel_size=2, padding='valid', strides=1 ,input_shape = (x_train.shape[1],1), activation='relu'))
# model.add(Conv1D(filters = 520, kernel_size=2, padding='same', strides=1 ,input_shape = (x_train.shape[1],1), activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
# model.add(Conv1D(filters = 50, strides=1 ,padding='same', kernel_size=3, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv1D(filters = 180, strides=1 ,padding='same', kernel_size=3, activation='relu'))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1))

#3.
path = './samsung/modelcheckpoint/sam4_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
cb = EarlyStopping(monitor='loss', patience=20, mode='auto')

opti = Adam(0.0005)
model.compile(loss=Huber(), optimizer=opti, metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=2000, batch_size=20, callbacks=[cb, mc], validation_data = (x_val, y_val))

model.save('./samsung/h5/sam4_model.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=20)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))


x_pred = x[-6:,:]
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
print(model.predict(x_pred).reshape(-1,))
print(y[-5:])

# 예측값, 실제값 시각화
# pred = model.predict(x_test[20:])
# plt.figure(figsize=(12, 9))
# plt.plot(np.asarray(y_test)[20:], label='actual')
# plt.plot(pred, label='prediction')
# plt.legend()
# plt.show()

# loss, val_loss 시각화
plt.figure(figsize=(12,9))      

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.legend()
plt.show()




