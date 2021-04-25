import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '../data/kaggle/titanic/propre_train.csv'
train_df = pd.read_csv(file_path, sep=',')

# print(train_df.info())
'''
Data columns (total 7 columns):
 #   Column    Non-Null Count   Dtype
---  ------    --------------   -----
 0   Survived  100000 non-null  int64
 1   Pclass    100000 non-null  int64
 2   Sex       100000 non-null  object
 3   Age       100000 non-null  float64
 4   SibSp     100000 non-null  int64
 5   Parch     100000 non-null  int64
 6   Embarked  100000 non-null  object
dtypes: float64(1), int64(4), object(2)
'''

#=====================관계 차트================
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['axes.grid'] = True 

def bar_chart(feature):
    survived = train_df[train_df['Survived']==1][feature].value_counts()
    dead = train_df[train_df['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True)
    plt.show()

# 등급(pclass)과의 관계
# print(bar_chart('Pclass')) # 1 class 살아남을 가능성 높아짐

# 성별(sex)와의 관계
# print(bar_chart('Sex')) # 여>남

# 연령대(age)와의 관계
train_df.loc[train_df['Age']<=20, 'Age']=2
train_df.loc[(train_df['Age']<=40)&(train_df['Age']>20), 'Age']=4
train_df.loc[(train_df['Age']<=60)&(train_df['Age']>40), 'Age']=6
train_df.loc[(train_df['Age']<=80)&(train_df['Age']>60), 'Age']=8
train_df.loc[train_df['Age']>80, 'Age']=10
# print(bar_chart('Age')) 


#=====================상관관계================
print(train_df.corr(method='pearson'))

#heatmap으로 상관관계를 표시
import seaborn as sb
plt.rcParams["figure.figsize"] = (5,5)
sb.heatmap(train_df.corr(),
           annot = True, #실제 값 화면에 나타내기
           cmap = 'Greens', #색상
           vmin = -1, vmax=1 , #컬러차트 영역 -1 ~ +1
          )
plt.show()