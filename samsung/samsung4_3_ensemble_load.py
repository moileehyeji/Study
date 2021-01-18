import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns


# 1. 데이터 
data_sam = np.load('./samsung/npy/samsung0115.npy')
data_inb = np.load('./samsung/npy/inbus.npy')

# print(data_sam.shape)       #(2399, 14)
# print(data_in.shape)        #(1088, 14)

data_sam = data_sam[1311:,[0,1,2,3,5,13]]        # 행 통일, 컬럼 슬라이싱       
data_inb = data_inb[:,[0,1,2,3,8,9]]             # 컬럼 슬라이싱                

# print(data_sam.shape)       #(1088, 6)
# print(data_inb.shape)       #(1088, 6)

# x, y  
x_sam = data_sam[:-3,:]         # samsung       시가 3일뒤까지 예측     
# print(x_sam.shape)              # (1085, 6)
x_inb = data_inb[:-3,:]         # inbus
# print(x_inb.shape)              # (1085, 6)

y_data = data_sam[1:,[0]]
y_data = y_data.reshape(-1,)
# print(y_data.shape)             #(1087,)
size = 3
def split_y(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)
y = split_y(y_data, size)

# print(y)                  #(1085, 3)


# 전처리 samsung
x_sam_train, x_sam_test, y_train, y_test = train_test_split(x_sam,y, test_size = 0.2, shuffle = True, random_state=110)
x_sam_train, x_sam_val, y_train, y_val = train_test_split(x_sam_train,y_train, test_size = 0.2, shuffle = True, random_state=110)

scaler1 = MinMaxScaler()
scaler1.fit(x_sam_train)
x_sam_train = scaler1.transform(x_sam_train)
x_sam_test = scaler1.transform(x_sam_test)
x_sam_val = scaler1.transform(x_sam_val)

# 차원변경
x_sam_train = x_sam_train.reshape(x_sam_train.shape[0],x_sam_train.shape[1],1)
x_sam_test = x_sam_test.reshape(x_sam_test.shape[0],x_sam_test.shape[1],1)
x_sam_val = x_sam_val.reshape(x_sam_val.shape[0],x_sam_val.shape[1],1)

# 전처리 inbus
x_inb_train, x_inb_test = train_test_split(x_inb, test_size = 0.2, shuffle = True, random_state=110)
x_inb_train, x_inb_val = train_test_split(x_inb_train, test_size = 0.2, shuffle = True, random_state=66)

scaler2 = MinMaxScaler()
scaler2.fit(x_inb_train)
x_inb_train = scaler2.transform(x_inb_train)
x_inb_test = scaler2.transform(x_inb_test)
x_inb_val = scaler2.transform(x_inb_val)

x_inb_train = x_inb_train.reshape(x_inb_train.shape[0],x_inb_train.shape[1],1)
x_inb_test = x_inb_test.reshape(x_inb_test.shape[0],x_inb_test.shape[1],1)
x__inb_val = x_inb_val.reshape(x_inb_val.shape[0],x_inb_val.shape[1],1)

# 2. 모델구성
model = load_model('./samsung/h5/sam6_model_8.h5')

#4. 평가, 예측
loss, mae = model.evaluate([x_sam_test, x_inb_test], y_test, batch_size=20)
print('loss, mae : ', loss, mae)


y_predict = model.predict([x_sam_test, x_inb_test])
from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))

x_pred1 = data_sam[-1:,:] 
x_pred2 = data_inb[-1:,:] 

x_pred1 = scaler1.transform(x_pred1)
x_pred2 = scaler2.transform(x_pred2)

x_pred1 = x_pred1.reshape(x_pred1.shape[0], x_pred1.shape[1], 1)
x_pred2 = x_pred2.reshape(x_pred2.shape[0], x_pred2.shape[1], 1)

y_pred = model.predict([x_pred1, x_pred2])
y_pred = y_pred.reshape(-1,)

print('월요일 시가: ', y_pred[0])
print('화요일 시가 : ', y_pred[1])




