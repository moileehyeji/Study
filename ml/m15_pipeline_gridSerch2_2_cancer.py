# Pipeline, make_pipeline
#                           + GridSearchCV

# 모델 비교

import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline        # concatenate와 Concatenate의 차이와 같음

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# parameters       
parameters1 = [ 
    {'mal__n_estimators':[100,200,300], 'mal__min_samples_split':[2,3,4,5], 'mal__n_jobs':[2,4]},  
    {'mal__n_estimators':[1,100],    'mal__max_depth':[35,40,44], 'mal__min_samples_leaf':[2,4,5], 'mal__min_samples_split':[8,10], 'mal__n_jobs':[3]},
    {'mal__n_estimators':[100,200], 'mal__min_samples_leaf':[12,24]},

]
parameters2 = [ 
    {'randomforestclassifier__n_estimators':[100,200,300], 'randomforestclassifier__min_samples_split':[2,3,4,5], 'randomforestclassifier__n_jobs':[2,4]},  
    {'randomforestclassifier__n_estimators':[1,100],    'randomforestclassifier__max_depth':[35,40,44], 'randomforestclassifier__min_samples_leaf':[2,4,5], 'randomforestclassifier__min_samples_split':[8,10], 'randomforestclassifier__n_jobs':[3]},
    {'randomforestclassifier__n_estimators':[100,200], 'randomforestclassifier__min_samples_leaf':[12,24]},

]

# 2. 모델구성       
# ====================================================================Pipeline + 
# Pipeline, make_pipeline : 전처리와 모델을 연결(통로)
# 별도 MinMaxScaler 필요없음

scalers = np.array([MinMaxScaler(), StandardScaler()])
for scaler in scalers:

    print('==========================',scaler)

    pipe =  Pipeline([('scaler', scaler), ('mal', RandomForestClassifier())])
    # pipe = make_pipeline(scaler, RandomForestClassifier())

    model = GridSearchCV(pipe, parameters1, cv=5)
    # model = RandomizedSearchCV(pipe, parameters1, cv=5)

    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가
    results = model.score(x_test, y_test)

    print('score       : ', results)   


'''
1. Tensorflow                 :
Dense(Dropout)모델 acc :  0.9912280440330505


2. RandomForest모델
============================================GridSearchCV
최종 정답률 :  0.9649122807017544
78.81초 걸렸습니다
============================================RandomizedSearchCV
최종 정답률 :  0.9649122807017544  
14.53초 걸렸습니다


2.  RandomForest모델, Pipeline()
========================== MinMaxScaler()
model_Pipeline의 score       :  0.9649122807017544
model_make_pipeline의 score  :  0.9649122807017544
========================== StandardScaler()       ****
model_Pipeline의 score       :  0.9736842105263158
model_make_pipeline의 score  :  0.9736842105263158


3. RandomForest모델, Pipeline, GridSearchCV
========================== MinMaxScaler()
score       :  0.9736842105263158
========================== StandardScaler()
score       :  0.9736842105263158
'''
