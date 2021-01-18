# hist를 이용하여 그래프를 그리시오
# loss, val_loss

#1.데이터
import numpy as np

#샘플데이터 로드
from sklearn.datasets import  load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# y=y.reshape(506,1)

# X_TRAIN 데이터 전처리 (MinMaxScaler)3 : 
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size = 0.8, shuffle = True, random_state = 150)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense = Dense(10, activation='relu')(input1)
dense = Dense(30, activation='relu')(dense)
dense = Dense(50, activation='relu')(dense)
dense = Dense(20, activation='relu')(dense)
output = Dense(1)(dense)
model = Model(inputs=input1, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#EarlyStopping정의 및 사용
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode='auto')
hist = model.fit(x_train, y_train, epochs=2000, batch_size=20, validation_data = (x_val, y_val), verbose=1, callbacks=[early_stopping])

# 그래프
import matplotlib.pyplot as plt

print(hist)
print(hist.history.keys())
print(hist.history)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss')
plt.ylabel('epochs')
plt.xlabel('loss')
plt.legend(['loss', 'val loss'])
plt.show()


# #4.평가, 예측
# loss, mae = model.evaluate(x_test, y_test, batch_size=1)
# print('loss : ', loss)
# print('mae : ',mae)
# y_predict = model.predict(x_test)

# from sklearn.metrics import mean_squared_error, r2_score
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('rmse : ', RMSE(y_test, y_predict))

# print('r2 : ', r2_score(y_test, y_predict))