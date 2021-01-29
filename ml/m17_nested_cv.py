# m12_gridSearch1 복사


import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 모델 import
from sklearn.svm import  LinearSVC, SVC

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# n등분 설정
kfold = KFold(n_splits=5, shuffle=True)


#   =========================================================================================================GridSearchCV
# parameters        : dictionary 형태      -->  GridSearchCV의 모델의 파라미터 튜닝
# GridSearchCV      : 이 자체가 모델
# best_estimator_   : 자동으로 최적의 가중치로 훈련한 예측값 출력

# parameters       
parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},                              # 4번 훈련
    {'C':[1, 10, 100],       'kernel':['rbf'],     'gamma':[0.001, 0.0001]},    # 6번 훈련
    {'C':[1, 10, 100, 100],  'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}     # 8번 훈련      : total 18번
]

#2. 모델구성
model = GridSearchCV(SVC(), parameters, cv=kfold)           # 5번 훈련      


score = cross_val_score(model, x_train, y_train, cv=kfold, verbose=1)  # 5번 훈련 * 5번 훈련  = 25번  

print('교차검증점수 : ', score)     # 교차검증점수 :  [0.95833333 0.91666667 0.91666667 1.         0.95833333]  -->cross_val_score의 점수

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

