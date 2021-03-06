# 특정 열이 전체 일정한 수로 예측
# 그 열의 quantile만 별도로 학습시 문제 없음
# 훈련할때마다 해당 열이 달라짐
# 해당열의 훈련중 loss값 거의 일정

# GHI 열 추가해보기
# df["Hour_Minute"] = df["Hour"] * 2 + df["Minute"] // 30

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

def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']+data['DHI'])
    return data

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
    
    data = Add_features(data)

    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]
    # temp = temp[['Hour','TARGET', 'DHI','DNI','WS','RH','T']]


    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')      # 다음날
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')    # 다다음날
        temp = temp.dropna()
        
        return temp.iloc[:-96]      # 이틀 제외 반환

    elif is_train==False:
        
        # temp = temp[['Hour','TARGET', 'DHI','DNI','WS','RH','T']]
        temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]
                              
        return temp.iloc[-48:, :]  


df_train = preprocess_data(train)



# =====================================================================데이터 X_test
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
print(X_test.shape)     #(27216, 7)  81일 

# =====================================================================상관계수
# print(df_train.corr())

# 상관계수 시각화
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(font_scale = 1.2)
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
# plt.show()


# =====================================================================데이터 split
def split_xy (data, time_steps):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps       # 1일차까지

        if x_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :-2]
        tmp_y = data[x_end_number-1: x_end_number, -2 : ]    # 7일차에서 target1

        x.append(tmp_x)
        y.append(tmp_y)   # Target1

    return np.array(x), np.array(y)





# =====================================================================데이터 전처리
df_train = df_train.to_numpy()
x, y = split_xy(df_train, 1)

X_test = X_test.to_numpy()

print('x : ', x.shape)      # (52464, 1, 7)
print('y : ', y.shape)      # (52464, 1, 2)


# 'Target1'
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 'Target2'
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2, random_state=0)

x_train = x_train.reshape(-1, x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[2])
x_val = x_val.reshape(-1, x_val.shape[2])
X_test = X_test.reshape(-1, X_test.shape[1])

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
X_test = scaler.transform(X_test)

x_train = x_train.reshape(-1, 1, 8)
x_test = x_test.reshape(-1, 1, 8)
x_val = x_val.reshape(-1, 1, 8)
X_test = X_test.reshape(-1, 1, 8)


# =====================================================================모델링
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]


def quantile_loss (q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

pred_list = []
loss_list = []

for q in quantiles : 

    # 1. 모델구성
    model = Sequential()
    # model.add(Conv1D(820,2,padding = 'same', activation = 'relu',input_shape = (x_train.shape[1], x_train.shape[2])))
    # model.add(Conv1D(650,2,padding = 'same', activation = 'relu',input_shape = (x_train.shape[1], x_train.shape[2])))
    # model.add(Conv1D(320,2,padding = 'same', activation = 'relu'))
    # model.add(Conv1D(120,2,padding = 'same', activation = 'relu'))
    # model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    # model.add(Flatten())    
    model.add(Dense(128, activation = 'relu',input_shape = (x_train.shape[1], x_train.shape[2])))
    # model.add(Dense(256, activation = 'relu'))
    # model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
    # model.add(Reshape((1,2)))
    model.add(Dense(2))

  

    #3. 컴파일, 훈련
    path = './dacon/modelcheckpoint/dacon_Dense_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    # mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

    opti = Adagrad()
    model.compile(loss=lambda y,x: quantile_loss(q,y,x), optimizer=opti , metrics='mae')
    model.fit(x_train, y_train, epochs=1000, batch_size=62, callbacks=[er, re], validation_data=(x_val, y_val), verbose=1)

    model.save('./dacon/h5/dacon6_2_unite_ghi.h5')
        
    # 4. 평가, 예측  
    # loss  
    loss = model.evaluate(x_test, y_test, batch_size=70)
    loss_list.append(loss[0])
    print(print('loss', q, ' : ', loss[0]))

    Y_pred = model.predict(X_test)
    # print(Y_pred.shape)     #(3888, 2)
    
    Y_pred = Y_pred.reshape(3888,2)
    Y_pred = pd.DataFrame(Y_pred) 
    Y_pred = pd.concat([Y_pred], axis=1)
    Y_pred[Y_pred<0] = 0
    Y_pred = Y_pred.to_numpy()
    # print(Y_pred.shape) #(3888, 2)

    # submission
    # column_name = 'q_' + str(q)
    column_name = f'q_{q}'
    submission.loc[submission.id.str.contains("Day7"), column_name] = Y_pred[:, 0].round(2)  # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = Y_pred[:, 1].round(2)   # Day8 (3888, 9)
    submission.to_csv(f'./dacon/submission/submission0124_6_0.7/submission_6_1_{q}.csv', index = False)  

   


loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
print('loss_mean : ', loss_mean)  

# to csv
submission.to_csv('./dacon/submission/submission0124_6_ghi/submission_6_2.csv', index = False)  


# submission.to_csv('./dacon/submission/unite_submission_7days.csv', index=True, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_1day.csv', index=True, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_1day_analysis1.csv', index=False, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_30minute_6.csv', index=False, encoding='cp949')        # score :1.93339
# submission.to_csv('./dacon/submission/unite_submission_30minute_6_0.7.csv', index=False, encoding='cp949')    # score :2.4791912422
submission.to_csv('./dacon/submission/unite_submission_30minute_6_GHI.csv', index=False, encoding='cp949')      # score :3.4142240883



'''
loss_mean :  2.9570015337732105
'''