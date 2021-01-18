#dataset -DateFrame - to_csv

import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset['data']
y = dataset['target']

# DataFrame
df = pd.DataFrame(x, columns=dataset['feature_names'])

# 컬럼명 수정
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# y컬럼 추가
df['Target'] = y

# to_csv
df.to_csv('../data/csv/iris_sklearn.csv', sep=',')

