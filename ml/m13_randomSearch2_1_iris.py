# RandomizedSearch : 특정모델 파라미터 자동화 (파라미터를 랜덤으로 쓰겠다)
# 빠름

# 실습 : RandomizedSearch 모델비교
#        작동시간 확인
'''
1.
from time import time
start = time()
finish = time()
print(finish - start)                                       #78.75598192214966

2.
import timeit
start_time = timeit.default_timer() # 시작 시간 체크
terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))    #78.663241초 걸렸습니다.  

3. 
import datetime as dt
before = dt.datetime.now()
after = dt.datetime.now()
print(after - before)       # 0:01:14.349831
'''


import datetime as dt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from time import time

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
# dataset = load_iris()
# x = dataset.data
# y = dataset.target

# csv 읽어오기
import pandas as pd
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# print(x.shape, y.shape)     #(150, 4) (150,)


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
# GridSearchCV : 이 자체가 모델
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)      


#3. 훈련
# 훈련시간 
start_time = time()

model.fit(x_train, y_train)

finish_time = time()

#4. 평가, 예측
# best_estimator_
print('최적의 매개변수 : ', model.best_estimator_)          # 최적의 매개변수 :  RandomForestClassifier(max_depth=44, min_samples_leaf=5, min_samples_split=10,n_estimators=1, n_jobs=3)   

y_pred = model.predict(x_test)                          
print('최종 정답률 : ', accuracy_score(y_test, y_pred))     # 최종 정답률 :  0.9666666666666667

aaa = model.score(x_test, y_test)
print(aaa)                                                 # 0.9666666666666667

#   =========================================================================================================

print(f"{finish_time - start_time:.2f}초 걸렸습니다")       # 74.97 --> 9.55초 걸렸습니다

'''
1. Tensorflow                 :    이게 이겨야 돼
Dense, LSTM, Conv1D 모델 acc :  1.0


2. RandomForest모델 :
============================================GridSearchCV
최종 정답률 :  1.0
74.97초 걸렸습니다
============================================RandomizedSearchCV
최종 정답률 :  0.9666666666666667
74.97 --> 9.55초 걸렸습니다


'''