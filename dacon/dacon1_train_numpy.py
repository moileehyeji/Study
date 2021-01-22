import numpy as np
import pandas as pd

data = pd.read_csv('./dacon/data/train/train.csv', header=0, index_col=[0,2])
data = data.astype('float64')


data_numpy = data.to_numpy()

np.save('./dacon/npy/dacon_train.npy', arr=data_numpy)
data.to_csv('./dacon/csv/dacon_train.csv', index=True, encoding='cp949')


# # shift

# data['TARGET1'] = data['TARGET'].shift(-48).fillna(method = 'ffill')             # 1일 뒤
# data['TARGET2'] = data['TARGET'].shift(-(48*2)).fillna(method = 'ffill')         # 2일 뒤

# print(data.columns)
# print(data.info())
# print(data.shape)   #(52560, 9)


# data_numpy = data.to_numpy()
# np.save('./dacon/npy/dacon_train_shift.npy', arr = data_numpy)
# data.to_csv('./dacon/csv/dacon_train_shift.csv', index=True, encoding='cp949')

