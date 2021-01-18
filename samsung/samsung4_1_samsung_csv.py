import pandas as pd
import numpy as np

df1 = pd.read_csv('./samsung/csv/samsung_1+2.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df2 = pd.read_csv('./samsung/csv/samsung3.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df1 = df1.drop(['2021-01-14'])              # 중복행 삭제
df2 = df2.loc['2021/01/15':'2021/01/14',:]  # 불필요 행 삭제
df2.index = ['2021-01-15','2021-01-14']     # 인덱스 수정

df = pd.concat([df2 ,df1])                                                      # 병합
df = df.sort_index(ascending=True)                                              # 오름차순
df = df.drop(columns = ['전일비', 'Unnamed: 6'])                                # 결측치 열삭제
df = df.astype('float64')

print(df.shape)         #(2399, 14)

data = df.to_numpy()

np.save('./samsung/npy/samsung0115.npy', arr=data)
df.to_csv('./samsung/csv/samsung_1+2+3.csv', index=True, encoding='cp949')


# 상관계수  : 0,1,2,3,5,6,7,13
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(font_scale = 1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

df.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
print(df.corr())