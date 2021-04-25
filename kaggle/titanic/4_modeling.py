import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_path = '../data/kaggle/titanic/propre_train2.csv'
test_path = '../data/kaggle/titanic/propre_test2.csv'
train = pd.read_csv(train_path, sep=',').astype('float32')
test = pd.read_csv(test_path, sep=',').astype('float32')

train['Fare'].fillna(0, inplace=True)
test['Fare'].fillna(0, inplace=True)

# print(train.info())
# print(test.info())
# print(train.isnull().sum())
# print(test.isnull().sum())

#데이터 
train_data = train.drop(['PassengerId','Survived'], axis=1)
target = train['Survived']

from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
clf = SVC(verbose=True)
# clf = DecisionTreeClassifier()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('../data/kaggle/titanic/submission_test1.csv', index=False)