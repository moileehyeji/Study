# keras33_LSTM1_boston1_sklearn -> Conv1D

#1.데이터
import numpy as np

#샘플데이터 로드
x = np.load('../data/npy/boston_x.npy')
y = np.load('../data/npy/boston_y.npy')

# x 데이터 전처리 (MinMaxScaler) : 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train , y_train ,train_size=0.8, shuffle = True, random_state = 66)

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

print(x.shape)  #(506, 13, 1)
print(y.shape)  #(506,)

# y=y.reshape(506,1)


#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout

input1 = Input(shape=(x.shape[1],1))
dense = Conv1D(filters = 100, kernel_size=2, padding='same', strides=1)(input1)
dense = MaxPooling1D(pool_size=2)(dense)
dense = Dropout(0.2)(dense)
# dense = Conv1D(filters = 50, kernel_size=3, padding='valid', strides=2)(dense)
# dense = MaxPooling1D(pool_size=2)(dense)
# dense = Dropout(0.2)(dense)
dense = Flatten()(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(20, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(80, activation='relu')(dense)
dense = Dropout(0.2)(dense)
dense = Dense(384, activation='relu')(dense)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)
model = Model(inputs=input1, outputs=output)


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#EarlyStopping정의 및 사용
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k54_2_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='auto')
model.fit(x_train,y_train, epochs=2000, batch_size=50, validation_data=(x_val, y_val), callbacks=[early_stopping, mc]) 

#4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse : ', RMSE(y_test, y_predict))

print('r2 : ', r2_score(y_test, y_predict))

''' 
Dense모델 : 
loss :  7.246347427368164
mae :  2.1537396907806396
rmse :  2.6919046845498684
r2 :  0.9133034676451246


LSTM모델 sklearn : 
loss :  8.797823905944824
mae :  2.25276255607605
rmse :  2.966112897719301
r2 :  0.8947413670437011

Conv1D모델 : 
loss :  7.341977596282959
mae :  2.019075632095337
rmse :  2.7096086975592613
r2 :  0.91215935299397694
'''
 
