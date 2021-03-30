import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats

num = 22
x = []
for i in range(num):           # 파일의 갯수
    # if i != 10:              # 10번파일은 빼고 확인해보겠다.
    # df = pd.read_csv(f'../../data/image/add/answer ({i}).csv', index_col=0, header=0)
    # df = pd.read_csv(f'C:/data/lotte/mode_csv/answer{i}.csv', index_col=0, header=0)
    df = pd.read_csv(f'C:/data/lotte/mode_csv/answer_add/answer_add_all/answer{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

# print(x.shape)
a= []
# df = pd.read_csv(f'../../data/image/add/answer ({i}).csv', index_col=0, header=0)
# df = pd.read_csv(f'C:/data/lotte/mode_csv/answer{i}.csv', index_col=0, header=0)
df = pd.read_csv(f'C:/data/lotte/mode_csv/answer_add/answer_add_all/answer{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(num):         # 파일의 갯수
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
        # ModeResult(mode=array([903]), count=array([12]))
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('C:/data/lotte/csv/sample.csv')
sub['prediction'] = np.array(a)
sub.to_csv(f'C:/data/lotte/mode_csv/answer_add/answer_add_all/answer_add_all{num}.csv',index=False)

# 77=> 6 => 80.419점 / 8 => 79점 / 10 => 80.926 / 12 => 81.893 / 13 => 82.251 / 14 => 82.374 / 17 => 83.047 / 21 => 83.568



