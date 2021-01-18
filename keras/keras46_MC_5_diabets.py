import numpy as np

#1. 데이터
from sklearn.datasets import load_diabetes

dataset =load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 110)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False, random_state = 100)


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

# y=y.reshape(442,1)
input1 = Input(shape=(10,))
dense = Dense(10)(input1)
dense = Dense(80)(dense)
dense = Dense(100)(dense)
dense = Dense(384)(dense)
dense = Dense(200)(dense)
# dense = Dense(10)(dense)
output = Dense(1)(dense)
model = Model(inputs=input1, outputs=output) 


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k45_diabets_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_data = (x_val, y_val), callbacks=[early_stopping, mc])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))

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
diabets5 결과:
loss, mae :  2166.6650390625 38.769779205322266
RMSE :  46.54745007951129
R2 :  0.6297812819678937

'''