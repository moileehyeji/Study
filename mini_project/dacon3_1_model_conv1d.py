import numpy as np

#1. 데이터
x = np.load('./dacon/npy/dacon_train_x.npy')
y = np.load('./dacon/npy/dacon_train_y.npy')

print(x.shape)  #(52129, 336, 7)
print(y.shape)  #(52129, 96)


# 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

print(x_train.shape)    #(33362, 336, 7)
print(x_test.shape)     #(10426, 336, 7)
print(x_val.shape)      #(8341, 336, 7)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters = 200, kernel_size=2, input_shape = (x_train.shape[1], x_train.shape[2]), strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters = 100, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(96))

# model.summary()

#3. 컴파일, 훈련, 평가
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, SGD, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

opti = np.array(['Adam'])   #,'Adamax','Adadelta', 'Adagrad', 'SGD', 'Nadam', 'RMSprop'])
path = './dacon/modelcheckpoint/dacon_conv1d_{val_loss:.4f}.hdf5'
er = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
re = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

loss = []
for i in opti:
    model.compile(loss = 'mse', optimizer= str(i) , metrics='mae')
    model.fit(x_train, y_train, epochs=2000, batch_size=100, callbacks=[er, re, mo], validation_data=(x_val, y_val), verbose=1)
    loss.append(model.evaluate(x_test, y_test, batch_size=100))

losss = np.array(loss) 
  
# print('Adadelta : ', losss[0])
# print('Adagrad  : ', losss[1])
# print('Adamax   : ', losss[0])
print('Adam     : ', losss[0])
# print('SGD      : ', losss[4])
# print('Nadam    : ', losss[5])
# print('RMSprop  : ', losss[6])


#4. 예측    :   검증용 test_x1,y1 사용
x_predic = np.load('./dacon/npy/dacon_test_x1.npy')
y_actual = np.load('./dacon/npy/dacon_test_y1.npy')

print(x_predic.shape)   #(80, 336, 7)

y_predic = model.predict(x_predic)
# print(y_predic) 
# print(y_predic.shape)   #(80, 96)
# print(y_actual)
# print(y_actual.shape)   #(80, 96)

y_predic = y_predic.reshape(1,-1) 
y_actual = y_actual.reshape(1,-1)
print('y_predic     : ', y_predic[:1, :20])
print('y_actual     : ', y_actual[:1, :20])



'''
y_predic = y_predic.reshape(-1,1)  
print(y_predic.shape)     #(7680, 1)  

# predict 값 to_csv
import pandas as pd
df = pd.DataFrame(y_predic)
print(df.shape)

df.to_csv('./dacon/csv/dacon_predict(7680, 1).csv')

'''

'''
훈련시 error : 
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_train_batch_end` time: 0.0080s). Check your callbacks.
-->verbose=1

sample shape (7776, 9) 맞추기
--> shift?
-->   if x_end_number > len(data): break?

'''




'''
Conv1D 모델, epochs = 1000 : 
                loss        mae
Adadelta :  [62.97805023  3.55808473]
Adagrad  :  [60.75506973  3.4456749 ]
Adamax   :  [59.0528717   3.38125253]   ***
Adam     :  [59.98241425  3.67598581]   ***
SGD      :  [665.30780029  21.30215454]
Nadam    :  [665.30413818  21.29043579]
RMSprop  :  [665.30291748  21.2810173 ]   

Adadelta :  [121.40640259   6.61423969]
Adagrad  :  [90.05518341  5.41335154]
Adamax   :  [43.74377441  3.48010159]   ***
Adam     :  [41.22418976  3.343853  ]   ***

Conv1D 모델, epochs = 2000 : 
Adamax   :  [44.3637619   3.52851152]
Adam     :  [39.59937668  3.18568778]   ***

Adam     :  [47.89339828  3.83865833]

'''

    



