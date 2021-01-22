import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import random
import warnings

warnings.filterwarnings(action='ignore')     #warning무시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, MaxPooling1D, Reshape
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, SGD, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import mean, maximum

# =====================================================데이터 train, submission

train = pd.read_csv('./dacon/data/train/train.csv')
submission = pd.read_csv('./dacon/data/sample_submission.csv')
# print('train : ', train.shape)      #train  :  (52560, 9)   1095일



# ======================================================중복값 찾기
# print(test.duplicated())

# # 중복값 몇개인지                
# print(test.duplicated().sum())    #43

# # 중복된 행의 데이터만 표시하기
# print(test[test.duplicated()])



# =====================================================================데이터 전처리 df_train
def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')      # 다음날
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')    # 다다음날
        temp = temp.dropna()
        
        return temp.iloc[:-96]      # 이틀 제외 반환

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]   # 이틀 반환 (예측값)


df_train = preprocess_data(train)
# print(df_train.iloc[:48])       #    Hour     TARGET  DHI  DNI   WS     RH   T    Target1    Target2



# =====================================================================데이터 X_test
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
# print(X_test.shape)     #(27216, 7)  81일 


# =====================================================================데이터 split
def split_xy (data, time_steps, y_row):
    
    x, y1, y2 = [], [], []

    for i in range(len(data)):

        x_end_number = i + time_steps       # 1일차까지
        y_end_number = x_end_number         # 1일차에서 target1, 2

        if y_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :-2]
        tmp_y1 = data[x_end_number - 48 : y_end_number, -2]    # 7일차에서 target1
        tmp_y2 = data[x_end_number - 48 : y_end_number, -1]    # 7일차에서 target2

        x.append(tmp_x)
        y1.append(tmp_y1)   # Target1
        y2.append(tmp_y2)   # Target2

    return np.array(x), np.array(y1), np.array(y2)





# =====================================================================데이터 전처리
df_train = df_train.to_numpy()
x, y1, y2 = split_xy(df_train, 24*2, 24*2)

X_test = X_test.to_numpy()

# print('x : ', x.shape)      # (52417, 48, 7)
# print('y1 : ', y1.shape)    # (52417, 48)
# print('y2 : ', y2.shape)    # (52417, 48)

# 'Target1'
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, test_size=0.2, random_state=0)
# 'Target2'
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x_train, y1_train, y2_train, test_size=0.2, random_state=0)

# print(x_train.shape)    # (33546, 48, 7)
# print(x_test.shape)     # (10484, 48, 7)
# print(x_val.shape)      # (8387, 48, 7)
# print(y1_train.shape)   # (33546, 48)
# print(y1_test.shape)    # (10484, 48)
# print(y1_val.shape)     # (8387, 48)
# print(y2_train.shape)   # (33546, 48)
# print(y2_test.shape)    # (10484, 48)
# print(y2_val.shape)     # (8387, 48)

x_train = x_train.reshape(-1, 48*7)
x_test = x_test.reshape(-1,48*7)
x_val = x_val.reshape(-1,48*7)
X_test = X_test.reshape(-1, 48*7)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
X_test = scaler.transform(X_test)

x_train = x_train.reshape(-1, 48, 7)
x_test = x_test.reshape(-1, 48, 7)
x_val = x_val.reshape(-1, 48, 7)
X_test = X_test.reshape(-1, 48, 7)


# =====================================================================모델링
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss (q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

def train_data(x_train, x_test, x_val, y_train, y_test, y_val, X_test):

    pred_list = []
    loss_list = []

    for q in quantiles : 

        # 1. 모델구성
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
        model.add(Dense(48, activation = 'relu'))
        model.add(Reshape((48,1)))
        model.add(Dense(1))

        #3. 컴파일, 훈련
        path = './dacon/modelcheckpoint/dacon_conv1d_{val_loss:.4f}.hdf5'
        er = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        re = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, verbose=1)
        # mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

        opti = Adam()
        model.compile(loss=lambda y,x: quantile_loss(q,y,x), optimizer=opti , metrics='mae')
        model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[er, re], validation_data=(x_val, y_val), verbose=1)
        
        # 4. 평가, 예측  
        # loss  
        loss = model.evaluate(x_test, y_test, batch_size=100)
        loss_list.append(loss)
        print(print('loss', q, ' : ', loss))

        Y_pred = model.predict(X_test)
        Y_pred = abs(Y_pred.round(2))
        print('y_pred.shape : ', Y_pred.shape)  #(81, 48, 1)
        pred_list.append(Y_pred)
        
    loss_list = np.array(loss_list)
    loss_list = loss_list.reshape(9,-1)

    pred_list = np.array(pred_list)
    pred_list = pred_list.reshape(3888,-1)

    return loss_list, pred_list


# target1 다음날
loss1, pred1 = train_data(x_train, x_test, x_val, y1_train, y1_test, y1_val, X_test)
# target2 다다음날
loss2, pred2 = train_data(x_train, x_test, x_val, y2_train, y2_test, y2_val, X_test)

print('loss1 : \n', loss1)
# print('loss2 : \n', loss2)


submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = pred1
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = pred2


# submission.to_csv('./dacon/submission/unite_submission_7days.csv', index=True, encoding='cp949')
submission.to_csv('./dacon/submission/unite_submission_1day.csv', index=True, encoding='cp949')