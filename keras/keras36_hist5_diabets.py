# hist를 이용하여 그래프를 그리시오
# loss, val_loss

import numpy as np

#1. 데이터
from sklearn.datasets import load_diabetes

dataset =load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.5, shuffle = False, random_state = 30)


#데이터 전처리3
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
dense = Dense(50, activation='linear')(input1)
dense = Dense(90, activation='linear')(dense)
# dense = Dense(100, activation='linear')(dense)
# dense = Dense(100, activation='linear')(dense)
# dense = Dense(100, activation='linear')(dense)
dense = Dense(80, activation='linear')(dense)
dense = Dense(80, activation='linear')(dense)
dense = Dense(80, activation='linear')(dense)
dense = Dense(20, activation='linear')(dense)
output = Dense(1)(dense)
model = Model(inputs=input1, outputs=output) 


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=2000, batch_size=50, validation_data = (x_val, y_val), callbacks=[early_stopping])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss')
plt.ylabel('epochs')
plt.xlabel('loss')
plt.legend(['loss', 'val loss'])
plt.show()


# #4. 평가, 예측
# loss, mae = model.evaluate(x_test, y_test, batch_size=10)
# print('loss, mae : ', loss, mae)

# y_predict = model.predict(x_test)

# from sklearn.metrics import mean_squared_error, r2_score
# def RMSE (y_test, y_predict):
#     return(np.sqrt(mean_squared_error(y_test, y_predict)))
# print('RMSE : ', RMSE(y_test, y_predict))
# print('R2 : ', r2_score(y_test, y_predict))