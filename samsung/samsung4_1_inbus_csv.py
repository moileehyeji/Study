import pandas as pd
import numpy as np

df = pd.read_csv('./samsung/csv/inbus.csv', index_col=0, header=0, encoding='cp949', thousands=',')

df = df.sort_index(ascending=True)                                             # 오름차순
df = df.drop(columns = ['전일비', 'Unnamed: 6'])                                # 결측치 열삭제  
df = df.astype('float64')

print(df)
print(df.info())
print(df.shape)     #(1088, 14)

data = df.to_numpy()

np.save('./samsung/npy/inbus.npy', arr=data)
df.to_csv('./samsung/csv/inbus_1.csv', index=True, encoding='cp949')

# 상관계수  : 0,1,2,3,4,9,11
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(font_scale = 1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

df.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
print(df.corr())