# RandomFrest 모델
# 파이프라인 역시 25번 돌리기
# 데이터는 wine


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

# n등분 설정
kfold = KFold(n_splits=5, shuffle=False) 

# ================================================================================================KFold.split
# split(X [, y, 그룹])  : 데이터를 학습 및 테스트 세트로 분할하는 인덱스를 생성
scores = list()
for train_index, test_index in kfold.split(x):
    print('================================================================================')
    print("TRAIN:", train_index, "\nTEST:", test_index) 

    # train : test
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]
      
    # train : test : validation
    for train_index, val_index in kfold.split(x_train):
        print("TRAIN:", train_index, "\nVAL:", val_index) 
        
        x_train, x_val = x[train_index], x[val_index] 
        y_train, y_val = y[train_index], y[val_index]

        print('x_train.shape : ', x_train.shape)        # (113, 13)
        print('x_test.shape  : ', x_test.shape)         # (36, 13)
        print('x_val.shape   : ', x_val.shape)          # (29, 13)

        print('y_train.shape : ', y_train.shape)        # (113,)
        print('y_test.shape  : ', y_test.shape)         # (36,)
        print('y_val.shape   : ', y_val.shape)          # (29,)


        # 훈련마다 평가
        #2. 모델구성
        model = RandomForestClassifier()

        score = cross_val_score(model, x_train, y_train, cv=kfold)     
        print('scores : ', score) 
        scores.append(score) 
                           
scores = np.array(scores)  
print(scores.shape)     # (25, 5)                                       

           
