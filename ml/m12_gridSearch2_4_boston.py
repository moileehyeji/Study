# 실습 : RandomForestRegressor
'''
parameters = [
    {'n_estimators'     :[100,200]},
    {'max_depth'        :[6,8,10,12]},
    {'min_sample_leaf'  :[3,5,7,10]},
    {'min_sample_leaf'  :[2,3,5,10]},
    {'n_jobs'           :[-1]}
]
'''

#모델비교

# gridSearch : 특정모델 파라미터 자동화 (격자형으로 촘촘하게 파라미터를 다 쓰겠다)


import numpy as np
import warnings

warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
from time import time

# 모델 import
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
dataset = load_boston()
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
    {'n_estimators':[200,300], 'min_samples_split':[2,5], 'n_jobs':[2,4]},  
    {'n_estimators':[1,100],   'max_depth':[13,21], 'min_samples_leaf':[2,4,5], 'min_samples_split':[8,10], 'n_jobs':[3]},
    {'n_estimators':[100,200], 'min_samples_leaf':[12,24]},

]


#2. 모델구성
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)

#3. 훈련
# 훈련시간 
start_time = time()

model.fit(x_train, y_train)

finish_time = time()

#4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)      # 최적의 매개변수 :  RandomForestRegressor(n_estimators=200, n_jobs=4)

y_pred = model.predict(x_test)
print('최종 정답률 : ', r2_score(y_test, y_pred))       # 최종 정답률 :  0.8571954130553036

aaa = model.score(x_test, y_test)                      # 0.8571954130553036
print(aaa)

print(f"{finish_time - start_time:.2f}초 걸렸습니다")   # 34.47초 걸렸습니다
                     

'''

1. Tensorflow            :
CNN모델 r2 :  0.9462232137123261


2. RandomForest모델 :
============================================GridSearchCV
최종 정답률 :  0.8571954130553036
34.47초 걸렸습니다
'''
