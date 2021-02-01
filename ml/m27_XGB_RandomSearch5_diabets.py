
'''
parameter = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsample_bylevel': [0.6, 0.7, 0.9]}
]

성능 좋은 파라미터로 구성되어있으니 공부 필요
'''

from time import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRFRegressor

# 1. 데이터
dataset = load_diabetes()
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
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],  'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]

#2. 모델구성

# GridSearchCV : 이 자체가 모델
model = RandomizedSearchCV(XGBRFRegressor(use_label_encoder=False), parameters, cv=kfold)       #(모델, 파라미터, kfold)   

#3. 훈련
# 훈련시간
start_time = time()

model.fit(x_train, y_train, eval_metric='logloss')

finish_time = time()

#4. 평가, 예측
# best_estimator_
print('최적의 매개변수 : ', model.best_estimator_)        # 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)

y_pred = model.predict(x_test)                           # 자동으로 최적의 가중치로 훈련한 예측값 출력
print('최종 정답률 : ', r2_score(y_test, y_pred))         # 최종 정답률 :  1.0

aaa = model.score(x_test, y_test)
print(aaa)                                               # 1.0

print(f"{finish_time - start_time:.2f}초 걸렸습니다")     # 74.97초 걸렸습니다


'''
1. Tensorflow            :
Conv1D모델 r2 :  0.6436679568820876


2. RandomForest모델 :
============================================GridSearchCV
최종 정답률 :  0.37948266914013495
79.99초 걸렸습니다
============================================RandomizedSearchCV
최종 정답률 :  0.3944350572120543
11.45초 걸렸습니다


3. XGB 모델
============================================GridSearchCV
============================================RandomizedSearchCV
'''