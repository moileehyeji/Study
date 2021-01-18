#dataset -DateFrame - to_csv - read_csv

import numpy as np
import pandas as pd

# index_col = 0 : index를 데이터로 인식 X
# header = 0    : 파일의 첫 번째 줄에서 열 이름이 유추
df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col = 0, header = 0)

print(df)