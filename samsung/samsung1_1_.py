
import pandas as pd
import numpy as np
import seaborn as sns
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

df = pd.read_csv('./samsung/csv/samsung.csv', index_col=0, header=0, encoding='cp949', thousands=',')

df = df.astype('float32')
df.index = pd.to_datetime(df.index)       # index DatetimeIndex
df = df.sort_index(ascending=True)        # index 기준 오름차순
df.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
'''
print(df)
print('df.columns : ', df.columns)
print('df.index : ', df.index)

print('df.head() : ', df.head())
print('df.tail() : ', df.tail())

print(df.info())
print(df.describe())
print(df.shape)    #(2400, 14)
'''
# 시각화
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.6, font='Malgun Gothic', rc = {'axes.unicode_minus':False})
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

df_x = df.loc['2018-05-08':'2021-01-12',:]
df_y = df.loc['2018-05-09':,:]

df_x = df_x.iloc[:,[0,1,2,3, 9,10,11,12]]   #9,12
df_y = df_y.iloc[:,3,]
'''
print('x.head() : ', df_x.head())
print('x.tail() : ', df_x.tail())
print('y.head() : ', df_y.head())
print('y.tail() : ', df_y.tail())

print(df_x.shape)   #(661, 6)
print(df_y.shape)   #(661,)
'''
x = df_x.to_numpy()
y = df_y.to_numpy()

np.save('./samsung/npy/samsung_x.npy', arr=x)
np.save('./samsung/samsung_y.npy', arr=y)


# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True, random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.3, shuffle = True, random_state=110)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

# np.save('./samsung/npy/samsung_x_train.npy', arr=x_train)
# np.save('./samsung/npy/samsung_x_test.npy', arr=x_test)
# np.save('./samsung/npy/samsung_x_val.npy', arr=x_val)
# np.save('./samsung/npy/samsung_y_train.npy', arr=y_train)
# np.save('./samsung/npy/samsung_y_test.npy', arr=y_test)
# np.save('./samsung/npy/samsung_y_val.npy', arr=y_val)

#2.
model = Sequential()
model.add(Conv1D(filters = 458, kernel_size=2, padding='same', strides=1 ,input_shape = (x_train.shape[1],1), activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters = 900, strides=1 ,padding='same', kernel_size=2, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters = 320, strides=1 ,padding='same', kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters = 210, strides=1 ,padding='same', kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 180, strides=1 ,padding='same', kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 80, strides=1 ,padding='same', kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(LSTM(16, activation='tanh'))
model.add(Dense(66, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#3.
path = './samsung/modelcheckpoint/sam3_{epoch:02d}_{loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='loss', save_best_only=True, mode = 'auto')
cb = EarlyStopping(monitor='loss', patience=20, mode='auto')

opti = Adam(0.0005)
model.compile(loss=Huber(), optimizer=opti, metrics=['mae'])
model.fit(x_train, y_train, epochs=2000, batch_size=20, callbacks=[cb, mc], validation_data = (x_val, y_val))

model.save('./samsung/h5/sam3_model.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=20)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))


x_pred = df_x.iloc[-6:,:]
x_pred = x_pred.to_numpy()
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
print(' 88,800 91,000 90600, 89700, ? : \n' , model.predict(x_pred))

# 예측값, 실제값 시각화
pred = model.predict(x_test[20:])
plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()

