# Pipeline, make_pipeline + GridSearchCV

# scaler후 cv 했을 때 과적합되는 문제(train데이터 전체 fit이후 train:val분리)를 Pipeline이 해결

# 모델 비교

import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline        # concatenate와 Concatenate의 차이와 같음

# 모델 import
from sklearn.svm import  LinearSVC, SVC

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# parameters       
parameters1 = [
    {'svc__C':[1, 10, 100, 1000], 'svc__kernel':['linear']},                                   # 4번 훈련
    {'svc__C':[1, 10, 100],       'svc__kernel':['rbf'],     'svc__gamma':[0.001, 0.0001]},    # 6번 훈련
    {'svc__C':[1, 10, 100, 100],  'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}     # 8번 훈련      : total 18번
]

parameters2 = [
    {'mal__C':[1, 10, 100, 1000], 'mal__kernel':['linear']},                                   # 4번 훈련
    {'mal__C':[1, 10, 100],       'mal__kernel':['rbf'],     'mal__gamma':[0.001, 0.0001]},    # 6번 훈련
    {'mal__C':[1, 10, 100, 100],  'mal__kernel':['sigmoid'], 'mal__gamma':[0.001, 0.0001]}     # 8번 훈련      : total 18번
]

# 2. 모델구성       
# ====================================================================Pipeline, GridSearchCV
# Pipeline, make_pipeline : 전처리와 모델을 연결(통로)
# 별도 MinMaxScaler 필요없음
# scaler후 cv 했을 때 과적합되는 문제(train데이터 전체 fit이후 train:val분리)를 Pipeline이 해결

# parameters 이름 앞에 모델의 이름추가
# Pipeline      : 설정이름__
# make_pipeline : 모델이름 소문자__ 

#

scalers = np.array([MinMaxScaler(), StandardScaler()])
for scaler in scalers:

    print('==========================',scaler)

    pipe =  Pipeline([('scaler', scaler), ('mal', SVC())])      # parameters1           
    # pipe = make_pipeline(scaler, SVC())                       # parameters2                 

    model = GridSearchCV(pipe, parameters2, cv=5)

    # 3. 훈련
    pipe.fit(x_train, y_train)

    # 4. 평가
    results = pipe.score(x_test, y_test)

    print('score  : ', results)  






'''
1. Tensorflow                 :    이게 이겨야 돼
Dense, LSTM, Conv1D 모델 acc :  1.0


2. RandomForest모델
============================================GridSearchCV
최종 정답률 :  1.0
74.97초 걸렸습니다
============================================RandomizedSearchCV
최종 정답률 :  0.9666666666666667
74.97 --> 9.55초 걸렸습니다


3. SCV모델
========================== MinMaxScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
========================== StandardScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
'''