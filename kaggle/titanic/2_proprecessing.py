import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_path = '../data/kaggle/titanic/train.csv'
test_path = '../data/kaggle/titanic/test.csv'
train = pd.read_csv(train_path, sep=',')
test = pd.read_csv(test_path, sep=',')

# print(train)
# print(train.info())
'''
Data columns (total 12 columns):
 #   Column       Non-Null Count   Dtype
---  ------       --------------   -----
 0   PassengerId  100000 non-null  int64
 1   Survived     100000 non-null  int64    생존 유무
 2   Pclass       100000 non-null  int64    티켓 등급
 3   Name         100000 non-null  object   승객 이름
 4   Sex          100000 non-null  object   성별
 5   Age          96708 non-null   float64  나이
 6   SibSp        100000 non-null  int64    동승자 수(형제 또는 배우자)
 7   Parch        100000 non-null  int64    동승자 수(부모 또는 자녀)
 8   Ticket       95377 non-null   object   티켓 번호
 9   Fare         99866 non-null   float64  티켓 요금
 10  Cabin        32134 non-null   object   선실 번호
 11  Embarked     99750 non-null   object   탑승 장소(선착장)
dtypes: float64(2), int64(5), object(5)
'''


# nan값을 가진 사람들의 특징
age_nan_rows = train[train['Age'].isnull()]
# print(age_nan_rows.head())

# 성별을 0,1로 표시
from sklearn.preprocessing import LabelEncoder
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
test['Sex'] = LabelEncoder().fit_transform(test['Sex']) 
# print(train.head(10)) 
# print(test.head() )
 
# ### 이름의 뒷부분을 고려하기엔 케이스가 너무 많아진다. 이름에서 앞의 성만 따서 생각해보자.

 
train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train['Name'].unique()
test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test['Name'].unique()
# print(len(titles)) #3953
# print(len(test_titles)) #3570
 

# 같은 NAME별 나이의 중앙값으로 채우기
# 그래도 train/test, 33,32 결측값 발생
train['Age'].fillna(-1, inplace=True)
test['Age'].fillna(-1, inplace=True)
 
medians_train = dict()
medians_test = dict()
for title in titles:
    median_train = train.Age[(train["Age"] != -1) & (train['Name'] == title)].median()
    medians_train[title] = median_train 
for title in test_titles:
    median_test = test.Age[(test["Age"] != -1) & (test['Name'] == title)].median()
    medians_test[title] = median_test 
 
for index, row in train.iterrows():
    if row['Age'] == -1:
        train.loc[index, 'Age'] = medians_train[row['Name']] 
for index, row in test.iterrows():
    if row['Age'] == -1:
        test.loc[index, 'Age'] = medians_test[row['Name']]

#=============================================================
# 같은 SEX별 나이의 중앙값으로 채우기
train['Age'].fillna(-1, inplace=True)
test['Age'].fillna(-1, inplace=True)
 
medians_train = dict()
medians_test = dict()
titles = train['Sex'].unique()
for title in titles:
    median_train = train.Age[(train["Age"] != -1) & (train['Sex'] == title)].median()
    medians_train[title] = median_train 
for title in titles:
    median_test = test.Age[(test["Age"] != -1) & (test['Sex'] == title)].median()
    medians_test[title] = median_test 
 
for index, row in train.iterrows():
    if row['Age'] == -1:
        train.loc[index, 'Age'] = medians_train[row['Sex']] 
for index, row in test.iterrows():
    if row['Age'] == -1:
        test.loc[index, 'Age'] = medians_test[row['Sex']]

 

# # 나이를 세분화, 10살부터 10살 단위로 60살까지
train.loc[train['Age']<=10, 'Age']=0
train.loc[(train['Age']<=20)&(train['Age']>10), 'Age']=1
train.loc[(train['Age']<=30)&(train['Age']>20), 'Age']=2
train.loc[(train['Age']<=40)&(train['Age']>30), 'Age']=3
train.loc[(train['Age']<=50)&(train['Age']>40), 'Age']=4
train.loc[(train['Age']<=60)&(train['Age']>50), 'Age']=5
train.loc[(train['Age']<=70)&(train['Age']>60), 'Age']=6
train.loc[train['Age']>70, 'Age']=7

test.loc[test['Age']<=10, 'Age']=0
test.loc[(test['Age']<=20)&(test['Age']>10), 'Age']=1
test.loc[(test['Age']<=30)&(test['Age']>20), 'Age']=2
test.loc[(test['Age']<=40)&(test['Age']>30), 'Age']=3
test.loc[(test['Age']<=50)&(test['Age']>40), 'Age']=4
test.loc[(test['Age']<=60)&(test['Age']>50), 'Age']=5
test.loc[(test['Age']<=70)&(test['Age']>60), 'Age']=6
test.loc[test['Age']>70, 'Age']=7
# print(train['Age'].unique())
# print(test['Age'].unique())

#Embarked컬럼 s클래스
num=0
for i in train['Embarked'] :
    if type(i)==float :
        train['Embarked'][num]='S'
    num+=1
num=0
for i in test['Embarked'] :
    if type(i)==float :
        test['Embarked'][num]='S'
    num+=1
print(train.isnull().sum())
print(test.isnull().sum())

# 동승자 2컬럼 통합
train['Family']=train['SibSp']+train['Parch']
train=train.drop(['SibSp','Parch'], axis=1)
test['Family']=test['SibSp']+test['Parch']
test=test.drop(['SibSp','Parch'], axis=1)

#=============================================================
# 같은 Family별 요금의 중앙값으로 채우기
# train['Fare'].fillna(0, inplace=True)
# test['Fare'].fillna(0, inplace=True)


#라벨링
train['Name'] = LabelEncoder().fit_transform(train['Name'])
test['Name'] = LabelEncoder().fit_transform(test['Name']) 
train['Ticket'] = LabelEncoder().fit_transform(train['Ticket'])
test['Ticket'] = LabelEncoder().fit_transform(test['Ticket']) 
train['Cabin'] = LabelEncoder().fit_transform(train['Cabin'])
test['Cabin'] = LabelEncoder().fit_transform(test['Cabin']) 
train['Embarked'] = LabelEncoder().fit_transform(train['Embarked'])
test['Embarked'] = LabelEncoder().fit_transform(test['Embarked']) 

print(train.isnull().sum())
print(test.isnull().sum())

train.to_csv('../data/kaggle/titanic/propre_train2.csv', index=False)
test.to_csv('../data/kaggle/titanic/propre_test2.csv', index=False)


# #데이터 
# train_data = train.drop(['PassengerId','Survived'], axis=1)
# target = train['Survived']

# from sklearn.svm import SVC
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score

# k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
# clf = SVC(verbose=True)
# # clf = DecisionTreeClassifier()
# clf.fit(train_data, target)

# test_data = test.drop("PassengerId", axis=1).copy()
# prediction = clf.predict(test_data)
# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": prediction
#     })

# submission.to_csv('../data/kaggle/titanic/submission_test1.csv', index=False)