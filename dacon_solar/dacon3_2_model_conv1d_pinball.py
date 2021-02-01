import warnings
warnings.filterwarnings('ignore')       #warning무시

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, SGD, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import mean, maximum


# 함수정의
def Conv1D_model():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(96))
    return model

def quantile_loss (q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)


#1. 데이터 로드
x = np.load('./dacon/npy/dacon_train_x.npy')
y = np.load('./dacon/npy/dacon_train_y.npy')
x_pred = np.load('./dacon/npy/dacon_test_x1.npy')
x_pred_submit = np.load('./dacon/npy/dacon_test_submit_x2.npy')

print(x.shape)  #(52129, 336, 7)
print(y.shape)  #(52129, 96)


# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

# print(x_train.shape)    #(33362, 336, 7)
# print(x_test.shape)     #(10426, 336, 7)
# print(x_val.shape)      #(8341, 336, 7)
# print(y_train.shape)    #(33362, 96)
# print(y_test.shape)     #(10426, 96)
# print(y_val.shape)      #(8341, 96)

x_train = x_train.reshape(-1, 336*7)
x_test = x_test.reshape(-1, 336*7)
x_val = x_val.reshape(-1, 336*7)
x_pred_submit = x_pred_submit.reshape(-1, 336*7)
x_pred = x_pred.reshape(-1, 336*7)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)
x_pred_submit = scaler.transform(x_pred_submit)

x_train = x_train.reshape(-1, 336, 7)
x_test = x_test.reshape(-1, 336, 7)
x_val = x_val.reshape(-1, 336, 7)
x_pred = x_pred.reshape(-1, 336, 7)
x_pred_submit = x_pred_submit.reshape(-1, 336, 7)



# ===========================================================================pinball============================================================================
q_lst  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#  q = tf.constant(np.array([qs]), dtype = tf.float32)
y_pred_all = []


for q in q_lst :
    # q = tf.constant(np.array([q_lst]), dtype = tf.float32)
    model = Conv1D_model() 

    #3. 컴파일, 훈련
    path = './dacon/modelcheckpoint/dacon_conv1d_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    re = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
    # mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

    opti = Adam()
    model.compile(loss=lambda y,x: quantile_loss(q,y,x), optimizer=opti , metrics='mae')
    model.fit(x_train, y_train, epochs=1000, batch_size=200, callbacks=[er, re], validation_data=(x_val, y_val), verbose=1)
       
    # 4. 평가, 예측  
    # loss  
    loss = model.evaluate(x_test, y_test, batch_size=200)
    print('Adam', q, ' : ', loss)  

    y_pred = model.predict(x_pred_submit)
    # print(y_pred.shape)     #((81, 96))
    y_pred = y_pred.reshape(-1,)
    y_pred = pd.Series(y_pred.round(2))
    y_pred_all.append(y_pred)
    print(y_pred.shape)


    

#==============================================================================================================================================================


# loss  
# loss = model.evaluate(x_test, y_test, batch_size=100)
# print('Adam   : ', loss)  


# submission
submission = pd.read_csv('./dacon/data/sample_submission.csv',  header=0, index_col=0)
submission = submission.astype('float64')

y_pred_all = np.array(y_pred_all)
y_pred_all = np.transpose(y_pred_all)   #(7776, 9)
y_pred_all = pd.DataFrame(y_pred_all, index = submission.index, columns = submission.columns)
y_pred_all = y_pred_all.astype('float64')
y_pred_all[y_pred_all<0] = 0

print(y_pred_all)

y_pred_all.to_csv('./dacon/submission/model_conv1d_pinball.csv', index=True, encoding='cp949')
























'''
★ Conv1D 모델: 
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

★ Conv1D 모델, Pinball, epochs = 5 : 
Adamax   :  [15.74185467 18.0643158 ]
Adam     :  [15.1419239  18.17181587]



'''

    



