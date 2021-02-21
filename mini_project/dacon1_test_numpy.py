import numpy as np
import pandas as pd


file_name = './dacon/data/test/{}.csv'
data_list = []
for i in range(81):
    data_list.append(pd.read_csv(file_name.format(i), header=0, index_col=[0,2]))
data = pd.concat(data_list)

data = data.astype('float64')

# 중복값 찾기
print(data.duplicated())

# 중복값 몇개인지                
print(data.duplicated().sum())    #43

# 중복된 행의 데이터만 표시하기
print(data[data.duplicated()])

# print(data.columns)
# print(data.shape)   #(27216, 7)
# print(data.info())

# data_numpy = data.to_numpy()
# np.save('./dacon/npy/dacon_test.npy', arr = data_numpy)
# data.to_csv('./dacon/csv/dacon_test.csv', index=True, encoding='cp949')


# shift

# data['TARGET1'] = data['TARGET'].shift(-48).fillna(method = 'ffill')             # 1일 뒤
# data['TARGET2'] = data['TARGET'].shift(-(48*2)).fillna(method = 'ffill')         # 2일 뒤

# print(data.columns)
# print(data.shape)   #(27216, 9)
# print(data.info())


# data_numpy = data.to_numpy()
# np.save('./dacon/npy/dacon_test_shift.npy', arr = data_numpy)
# data.to_csv('./dacon/csv/dacon_test_shift.csv', index=True, encoding='cp949')
