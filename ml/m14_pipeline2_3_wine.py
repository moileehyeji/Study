# Pipeline, make_pipeline

# 모델 비교
# 2번부터 RandomForest 모델 사용

import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline        # concatenate와 Concatenate의 차이와 같음

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# Pipeline 사용시 필요 없음
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델구성       
# ====================================================================Pipeline
# Pipeline, make_pipeline : 전처리와 모델을 연결(통로)
# 별도 MinMaxScaler 필요없음
scalers = np.array([MinMaxScaler(), StandardScaler()])
for scaler in scalers:

    print('==========================',scaler)

    model_Pipeline =  Pipeline([('scaler', scaler), ('malddong', RandomForestClassifier())])
    model_make_pipeline = make_pipeline(scaler, RandomForestClassifier())

    # 3. 훈련
    model_Pipeline.fit(x_train, y_train)
    model_make_pipeline.fit(x_train, y_train)

    # 4. 평가
    results1 = model_Pipeline.score(x_test, y_test)
    results2 = model_make_pipeline.score(x_test, y_test)

    print('model_Pipeline의 score       : ', results1)     
    print('model_make_pipeline의 score  : ', results2)  


'''
1. Tensorflow                 :
Dense모델 acc :  1.0


2. RandomForest모델
============================================GridSearchCV
최종 정답률 :  1.0
31.17초 걸렸습니다

============================================RandomizedSearchCV
최종 정답률 :  1.0
18.34초 걸렸습니다


3. RandomForest모델, Pipeline()
========================== MinMaxScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
========================== StandardScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
'''
