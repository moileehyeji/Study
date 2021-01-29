# RandomizedSearch : 특정모델 파라미터 자동화 (파라미터를 랜덤으로 쓰겠다)
# 빠름

# 실습 : RandomizedSearch 모델비교
#        작동시간 확인


import datetime as dt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from time import time

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     #--> 이진분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_breast_cancer()
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
    {'n_estimators':[100,200,300], 'min_samples_split':[2,3,4,5], 'n_jobs':[2,4]},  
    {'n_estimators':[1,100],    'max_depth':[35,40,44], 'min_samples_leaf':[2,4,5], 'min_samples_split':[8,10], 'n_jobs':[3]},
    {'n_estimators':[100,200], 'min_samples_leaf':[12,24]},

]


#2. 모델구성
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)

#3. 훈련
# 훈련시간 
start_time = time()

model.fit(x_train, y_train)

finish_time = time()

#4. 예측, 평가
print("최적의 매개변수 : ", model.best_estimator_)          # 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=4)

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))     # 최종 정답률 :  0.9649122807017544  

print(f"{finish_time - start_time:.2f}초 걸렸습니다")       # 78.81 --> 14.53초 걸렸습니다

'''
Tensorflow                 :
Dense(Dropout)모델 acc :  0.9912280440330505
'''