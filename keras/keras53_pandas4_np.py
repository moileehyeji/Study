#dataset - DateFrame - to_csv - read_csv - numpy

import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col = 0, header = 0)
print(df)          #(150,5)
print(df.info())   # x: float64, y: int64

# pandas dataframe -> numpy 1
aaa = df.to_numpy()
print(aaa)        # target 값 float로 바뀜 (numpy 한가지 형태)
print(aaa.shape)  # (150, 5)
print(type(aaa))  # <class 'numpy.ndarray'>
# pandas dataframe -> numpy 2
bbb = df.values
print(bbb)        # aaa와 동일
print(type(bbb))  # <class 'numpy.ndarray'>

np.save('../data/npy/iris_sklearn.npy', arr=aaa)

# numpy 슬라이싱
x = aaa[:,:4]
y = aaa[:,4]
print(x)
print(y)