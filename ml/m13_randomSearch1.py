# m10_kfold_1 복사
# RandomizedSearch : 특정모델 파라미터 자동화 (파라미터를 랜덤으로 쓰겠다)
# 빠름


import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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


#   =========================================================================================================RandomizedSearchCV
# parameters        : dictionary 형태      -->  RandomizedSearchCV의 모델의 파라미터 튜닝
# RandomizedSearchCV: 이 자체가 모델
# RandomizedSearchCV의 파라미터 : n_iter는 몇 번 반복하여 수행할 것인지에 대한 값, default는 10이다
#                        -> verbose = 1 : Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_estimator_   : 자동으로 최적의 가중치로 훈련한 예측값 출력


# parameters       
parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},                              
    {'C':[1, 10, 100],       'kernel':['rbf'],     'gamma':[0.001, 0.0001]},    
    {'C':[1, 10, 100, 100],  'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}    
]

#2. 모델구성
# model = SVC()

# GridSearchCV : 이 자체가 모델
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)  #(모델, 파라미터, kfold)    : n_iter 기본값 10 * kfold 5분리 = 50


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# best_estimator_
print('최적의 매개변수 : ', model.best_estimator_)        # 최적의 매개변수 :  SVC(C=100, gamma=0.001)

y_pred = model.predict(x_test)                           # 자동으로 최적의 가중치로 훈련한 예측값 출력
print('최종 정답률 : ', accuracy_score(y_test, y_pred))   # 최종 정답률 :  1.0

aaa = model.score(x_test, y_test)
print(aaa)                                               # 1.0

#   =========================================================================================================

# scores = cross_val_score(model, x, y, cv=kfold)    
# print('scores : ', scores)    
           



'''
Tensorflow                 :    이게 이겨야 돼
Dense, LSTM, Conv1D 모델 acc :  1.0

'''