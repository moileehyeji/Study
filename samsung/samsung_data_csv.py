import pandas as pd
import numpy as np

df1 = pd.read_csv('./samsung/csv/samsung1.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df2 = pd.read_csv('./samsung/csv/samsung2.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df1 = df1.drop(['2021-01-13'])

df = pd.concat([df2 ,df1])                                                      # 병합
df = df.sort_index(ascending=True)                                              # 오름차순
df = df.drop(columns = ['전일비', 'Unnamed: 6'])                                # 결측치 열삭제      
df = df.dropna(axis=0)                                                          # 결측치 행삭제 : df.dropna(axis=0 : 결측치 행삭제, axis=1 : 결측치 열삭제 )       
df.loc[:'2018-04-27','시가':'종가'] = (df.loc[:'2018-04-27','시가':'종가'])/50.  # /50


data = df.to_numpy()

np.save('./samsung/npy/samsung0114_1.npy', arr=df)
df.to_csv('./samsung/csv/samsung_1+2.csv', index=True, encoding='cp949')

print(df1.index)