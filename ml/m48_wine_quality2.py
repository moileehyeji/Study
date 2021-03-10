import numpy as np
import pandas as pd

# csv불러오기
file_dir = 'C:/data/csv/wine'
wine = pd.read_csv(f'{file_dir}/winequality-white.csv', sep=';', index_col=None, header=0)
wine_test = pd.read_csv(f'{file_dir}/data-01-test-score.csv', sep=';', index_col=None, header=0)
# print(wine.shape)   #(4898, 12)
# print(wine.describe())


# pandas.groupby 
# 객체를 분할하는 기능을 적용하고, 그 결과 결합의 조합을 포함한다. 
# 이는 대량의 데이터를 그룹화하고 이러한 그룹에 대한 연산을 계산하는 데 사용할 수 있습니다.
# 전체 데이터를 그룹 별로 나누고 (split), 
# 각 그룹별로 집계함수를 적용(apply) 한후, 
# 그룹별 집계 결과를 하나로 합치는(combine) 단계를 거침
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

import matplotlib.pyplot as plt 
count_data.plot()
plt.show()


# category 즉, 데이터를 조절할 수 있는 권한이 있을때만 가능