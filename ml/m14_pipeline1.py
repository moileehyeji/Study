# Pipeline, make_pipeline

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

# Pipeline 사용시 필요 없음
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델구성       
# ====================================================================Pipeline
# Pipeline, make_pipeline : 전처리와 모델을 연결(통로)
# 별도 MinMaxScaler 필요없음
# scaler후 cv 했을 때 과적합되는 문제(train데이터 전체 fit이후 train:val분리)를 Pipeline이 해결

scalers = np.array([MinMaxScaler(), StandardScaler()])
for scaler in scalers:

    print('==========================',scaler)

    model_Pipeline =  Pipeline([('scaler', scaler), ('malddong', SVC())])
    model_make_pipeline = make_pipeline(scaler, SVC())

    # 3. 훈련
    model_Pipeline.fit(x_train, y_train)
    model_make_pipeline.fit(x_train, y_train)

    # 4. 평가
    results1 = model_Pipeline.score(x_test, y_test)
    results2 = model_make_pipeline.score(x_test, y_test)

    print('model_Pipeline의 score       : ', results1)     
    print('model_make_pipeline의 score  : ', results2)  

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


3. SCV모델, Pipeline
========================== MinMaxScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
========================== StandardScaler()
model_Pipeline의 score       :  1.0
model_make_pipeline의 score  :  1.0
'''