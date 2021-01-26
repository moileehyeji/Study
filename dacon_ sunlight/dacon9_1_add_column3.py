

# 특정 열이 전체 일정한 수로 예측
# 그 열의 quantile만 별도로 학습시 문제 없음
# 훈련할때마다 해당 열이 달라짐
# 해당열의 훈련중 loss값 거의 일정
# --> 해당 열 두번 훈련해보기 OOOOOOOOO
# --> Hour_Minute 컬럼 

# dacon7_1_unite_addmodel_submit 복사
# 컬럼 추가
# 모델, lr, fac 튜닝
# 컬럼 삭제


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



# =====================================================데이터 train(52560, 9), submission(7776, 10)

train = pd.read_csv('./dacon/data/train/train.csv')
submission = pd.read_csv('./dacon/data/sample_submission.csv')
# print('train : ', train.shape)    
# print('submission : ', submission.shape)



# ======================================================중복값 찾기
# print(test.duplicated())

# # 중복값 몇개인지                
# print(test.duplicated().sum())    #43

# # 중복된 행의 데이터만 표시하기
# print(test[test.duplicated()])

# =====================================================================함수 : 컬럼 추가

# 함수 : GHI column 추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data 
    
# 함수 : Hour_Minute column 추가
def Add_features_hour_minute(data):
    data.insert(0, 'Hour_Minute', data["Hour"] * 2 + data["Minute"] // 30)
    data.insert(0, 'ordering', (data["Day"] // 4))
    data.insert(0, 'Day_', data["Day"] % 4)
    return data 

# =====================================================================함수 : 모델

def mymodel1():
    # 1. 모델구성
    model = Sequential()
    # model.add(Dense(128, activation = 'relu',input_shape = (x_train.shape[1],x_train.shape[2])))
    model.add(Dense(128, activation = 'relu',input_dim = x_train.shape[1]))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
    # model.add(Reshape((1,2)))
    model.add(Dense(2))
    return model

def mymodel2():
    # 1. 모델구성
    model = Sequential()
    # model.add(Dense(128, activation = 'relu',input_shape = (x_train.shape[1],x_train.shape[2])))
    # model.add(Dense(128, activation = 'relu',input_dim = 8))
    # model.add(Dense(64, activation = 'relu'))
    # model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu', input_dim =  x_train.shape[1]))
    model.add(Dense(2, activation = 'relu'))
    # model.add(Reshape((1,2)))
    model.add(Dense(2))
    return model


# =====================================================================함수 : train(이틀제외), test(predict 할 하루씩) 컬럼자르기
def preprocess_data(data, is_train=True):
    
    data = Add_features(data)
    data = Add_features_hour_minute(data)
    data = data.astype('float64')
    temp = data.copy()
    # temp = temp[['Hour_Minute','TARGET','GHI','DHI','DNI','WS','RH','T']]
    # temp = temp[['Day_','ordering','Hour_Minute','TARGET','GHI','DHI','DNI','WS','RH','T']]
    temp = temp[['Day_','ordering','Hour_Minute','TARGET','GHI','DHI','DNI','WS','T']]


    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')      # 다음날
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')    # 다다음날
        temp = temp.dropna()
        
        return temp.iloc[:-96]      # 이틀 제외 반환

    elif is_train==False:
        
        # temp = temp[['Hour_Minute','TARGET','GHI','DHI','DNI','WS','RH','T']]
        # temp = temp[['Day_','ordering','Hour_Minute','TARGET','GHI','DHI','DNI','WS','RH','T']]
        temp = temp[['Day_','ordering','Hour_Minute','TARGET','GHI','DHI','DNI','WS','T']]

                              
        return temp.iloc[-48:, :]  


# =====================================================================상관계수
# df_train = preprocess_data(train)
# print(df_train.corr())

# # 상관계수 시각화
# import matplotlib.pyplot as plt 
# import seaborn as sns 
# sns.set(font_scale = 1.2)
# sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
# plt.show()


# =====================================================================함수 : split
def split_xy (data, time_steps):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps       # 1일차까지

        if x_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :-2]
        tmp_y = data[x_end_number-1:x_end_number, -2 : ]    # 7일차에서 target1

        x.append(tmp_x)
        y.append(tmp_y)   # Target1

    return np.array(x), np.array(y)


    
# =====================================================================데이터 X_test (48*81, 8)
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

df_test = pd.concat(df_test)
# print(df_test.shape)     #(3888, 7)  81일 
# X_test.to_csv('./dacon/csv/X_test2.csv', index=False, encoding='cp949')

# print(X_test.info())

print(df_test.columns)
print(df_test)

X_test = df_test.to_numpy()
# print('X_test   : ', X_test.shape)


# =====================================================================데이터 np_train (52560-96, 10)
df_train = preprocess_data(train)
# df_train.to_csv('./dacon/csv/df_train.csv', index=False, encoding='cp949')
# print(df_train.info())

print(df_train.columns)
print(df_train)

np_train = df_train.to_numpy()
# print('np_train : ' , np_train.shape)



# =====================================================================데이터 전처리

# x, y = split_xy(np_train, 1)

x = np_train[:,:-2]
y = np_train[:,-2:]

print('x : ', x)      # (52464, 11)
print('y : ', y)      # (52464, 2)

print(x[:1])
print(y[:1])


# 'Target1'
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 'Target2'
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# x_train = x_train.reshape(-1, x_train.shape[2])
# x_test = x_test.reshape(-1, x_test.shape[2])
# x_val = x_val.reshape(-1, x_val.shape[2])
# X_test = X_test.reshape(-1, X_test.shape[1])

print(x_train.shape)    #(33576, 10)
print(x_test.shape)     #(10493, 10)
print(x_val.shape)      #(8395, 10)
print(X_test.shape)     #(3888, 10)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
X_test = scaler.transform(X_test)

# x_train = x_train.reshape(-1, 1, 10)
# x_test = x_test.reshape(-1, 1, 10)
# x_val = x_val.reshape(-1, 1, 10)
# X_test = X_test.reshape(-1, 1, 10)



# =====================================================================모델링
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def quantile_loss (q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)


pred_list = []
loss_list = []

for q in quantiles : 

        #2. 모델
        model = mymodel1()

        #3. 컴파일, 훈련
        path = './dacon/modelcheckpoint/dacon_conv1d_{val_loss:.4f}.hdf5'
        er = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        re = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.3, verbose=1)
        # mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

        opti = Adam()
        model.compile(loss=lambda y,pred: quantile_loss(q, y, pred), optimizer=opti , metrics='mae')
        model.fit(x_train, y_train, epochs=1000, batch_size=20, callbacks=[er, re], validation_data=(x_val, y_val), verbose=1)
        
        # 4. 평가, 예측  
        # loss  
        loss = model.evaluate(x_test, y_test, batch_size=20)
        print(print('첫번째 loss', q, ' : ', loss[0]))

        Y_pred = model.predict(X_test)
        # print(Y_pred.shape)     #(3888, 2)
        
        Y_pred = Y_pred.reshape(3888,2)
        # Y_pred = pd.DataFrame(Y_pred) 
        # Y_pred = pd.concat([Y_pred], axis=1)
        # Y_pred[Y_pred<0] = 0
        # Y_pred = Y_pred.to_numpy()

        #===============================================문제컬럼 재훈련 1차==============================================================
        if (Y_pred[0:1, 0] == Y_pred[20:21, 0]):        # day7 00시와 09시가 같으면
            print('**********************************************************훈련다시 1차**********************************************************')
            print(Y_pred[0:1, 0],   Y_pred[20:21, 0])

            #2. 모델
            model = mymodel2()

            #3. 컴파일, 훈련
            path = './dacon/modelcheckpoint/dacon_conv1d_{val_loss:.4f}.hdf5'
            er = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
            re = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, verbose=1)
            # mo = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode='auto')

            opti = Adam(learning_rate=0.09)
            model.compile(loss=lambda y,pred: quantile_loss(q, y, pred), optimizer=opti , metrics='mae')
            model.fit(x_train, y_train, epochs=200, batch_size=60, callbacks=[er, re], validation_data=(x_val, y_val), verbose=1)
            
            # 4. 평가, 예측  
            # loss  
            loss = model.evaluate(x_test, y_test, batch_size=60)
            # loss_list.append(loss[0])
            print('두번째 loss', q, ' : ', loss[0])

            # Y_pred = model.predict(X_test)
            # # print(Y_pred.shape)     #(3888, 2)
            
            # Y_pred = Y_pred.reshape(3888,2)
            # Y_pred = pd.DataFrame(Y_pred) 
            # Y_pred = pd.concat([Y_pred], axis=1)
            # Y_pred[Y_pred<0] = 0
            # Y_pred = Y_pred.to_numpy()

        #=============================================================================================================

        # 최종 loss 
        loss_list.append(loss[0])

        #최종 predict
        Y_pred = model.predict(X_test)
        # print(Y_pred.shape)     #(3888, 2)
            
        Y_pred = Y_pred.reshape(3888,2)
        Y_pred = pd.DataFrame(Y_pred) 
        Y_pred = pd.concat([Y_pred], axis=1)
        Y_pred[Y_pred<0] = 0
        Y_pred = Y_pred.to_numpy()


        # submission
        # column_name = 'q_' + str(q)
        column_name = f'q_{q}'          #f string
        submission.loc[submission.id.str.contains("Day7"), column_name] = Y_pred[:, 0].round(2)  # Day7 (3888, 9)
        submission.loc[submission.id.str.contains("Day8"), column_name] = Y_pred[:, 1].round(2)   # Day8 (3888, 9)
        submission.to_csv(f'./dacon/submission/submission0126_add_column/submission_7_1_{q}.csv', index = False)  





loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
print('loss_mean : ', loss_mean)  

# to csv
submission.to_csv('./dacon/submission/submission0126_add_column/submission_7_2.csv', index = False)  # score :1.9498123625


# submission.to_csv('./dacon/submission/unite_submission_7days.csv', index=True, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_1day.csv', index=True, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_1day_analysis1.csv', index=False, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_30minute_6.csv', index=False, encoding='cp949')        # score :1.93339
# submission.to_csv('./dacon/submission/unite_submission_30minute_6_0.7.csv', index=False, encoding='cp949')    # score :2.4791912422
# submission.to_csv('./dacon/submission/unite_submission_30minute_6_GHI.csv', index=False, encoding='cp949')    # score :3.4142240883
# submission.to_csv('./dacon/submission/unite_submission_quantile.csv', index=False, encoding='cp949')
# submission.to_csv('./dacon/submission/unite_submission_addmodel.csv', index=False, encoding='cp949')         #score :1.9498123625
# submission.to_csv('./dacon/submission/unite_submission_addmodel_quantile.csv', index=False, encoding='cp949') #score :
# submission.to_csv('./dacon/submission/unite_submission_td.csv', index=False, encoding='cp949')                #score :1.9780125974	
# submission.to_csv('./dacon/submission/unite_submission_td_addmodel2.csv', index=False, encoding='cp949')        #score :2.0518367045
# submission.to_csv('./dacon/submission/unite_submission_add_column.csv', index=False, encoding='cp949')         #score :
# submission.to_csv('./dacon/submission/unite_submission_add_column2.csv', index=False, encoding='cp949')         #score :
submission.to_csv('./dacon/submission/unite_submission_add_column3.csv', index=False, encoding='cp949')         #score :



'''

# 1. 컬럼 추가          -> loss_mean :  1.9431294070349798     -> submission_add_column
# 2. 모델, lr, fac 튜닝 -> loss_mean :  2.0340065823660956     -> submission_add_column2
# 3. 컬럼 삭제          -> loss_mean :  1.9652686847580805     -> submission_add_column3
# 4. 모델 conv1d
'''