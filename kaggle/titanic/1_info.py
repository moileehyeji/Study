import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# print(train.isnull().sum())
'''
PassengerId        0
Survived           0
Pclass             0
Name               0
Sex                0
Age             3292    나이
SibSp              0
Parch              0
Ticket          4623    티켓 번호
Fare             134    티켓 요금
Cabin          67866    선실 번호
Embarked         250    탑승 장소(선착장)
'''

# print(test.isnull().sum())
'''
PassengerId        0
Pclass             0
Name               0
Sex                0
Age             3487
SibSp              0
Parch              0
Ticket          5181
Fare             133
Cabin          70831
Embarked         277
'''

print(train['Embarked'].value_counts())
print(test['Embarked'].value_counts())
'''
S    72139
C    22187
Q     5424
Name: Embarked, dtype: int64
S    68842
C    22308
Q     8573
Name: Embarked, dtype: int64
'''