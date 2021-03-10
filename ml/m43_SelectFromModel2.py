# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피쳐임포턴스 구할 것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피쳐갯수를 구할 것

# 3. 위 피쳐 개수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2구할 것

# 1, 2 비교


import numpy as np 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

# parameters       
parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],  'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]

x, y = load_boston(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

# model = XGBRegressor(n_jobs = 8)
model = GridSearchCV(XGBRegressor(n_jobs = 8), parameters)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('최적의 파라미터 : ', model.best_estimator_)
print('r2 : ', score)
# r2 :  0.9311752976588772 




model = model.best_estimator_   # 최적의 파라미터로 모델 생성
# ----------------------------------------------------------------중요도 오름차순
thresholds = np.sort(model.feature_importances_)    #오름차순
print(thresholds)
# [0.00152491 0.00226998 0.00988215 0.01005833 0.0132021  0.01668662
#  0.02487051 0.0313093  0.04498584 0.05127558 0.24242924 0.27001473
#  0.28149077]
print(np.sum(thresholds))   #1.0



# ----------------------------------------------------------------SelectFromModel
# 중요도 가중치를 기반으로 기능을 선택하기위한 메타 트랜스포머.
# Xgbooster, LGBM, RandomForest등 feature_importances_기능을 쓰는 모델이면 사용 가능
# prefit: True 인 경우 transform직접 호출 
# coef_?
""" for thresh in thresholds:
    selection = SelectFromModel(model, 
                                threshold = thresh, #Feature 선택에 사용할 임계 값, 중요도가 크거나 같은 기능은 유지되고 나머지는 삭제
                                prefit=True         #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                                )

    # x_train을 선택한 Feature로 줄입니다.
    selection_x_train = selection.transform(x_train)
    print(selection_x_train.shape)

    selection_model = GridSearchCV(XGBRegressor(n_jobs = 8), parameters)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, selection_x_train.shape[1], score*100))

# (404, 13)
# Thresh=0.002, n=13, R2:92.21%
# (404, 12)
# Thresh=0.002, n=12, R2:92.16%
# (404, 11)
# Thresh=0.010, n=11, R2:92.03%
# (404, 10)
# Thresh=0.010, n=10, R2:92.19%
# (404, 9)
# Thresh=0.013, n=9, R2:92.59%
# (404, 8)
# Thresh=0.017, n=8, R2:92.71%
# (404, 7)
# Thresh=0.025, n=7, R2:92.86%
# (404, 6)
# Thresh=0.031, n=6, R2:92.71%
# (404, 5)
# Thresh=0.045, n=5, R2:91.74%
# (404, 4)
# Thresh=0.051, n=4, R2:91.47%
# (404, 3)
# Thresh=0.242, n=3, R2:78.35%
# (404, 2)
# Thresh=0.270, n=2, R2:69.41%
# (404, 1)
# Thresh=0.281, n=1, R2:44.98%  
#    : 
#    :           
"""

selection = SelectFromModel(model, 
                            threshold = 0.025, #Feature 선택에 사용할 임계 값, 중요도가 크거나 같은 기능은 유지되고 나머지는 삭제
                            prefit=True        #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                            )

# x_train을 선택한 Feature로 줄입니다.
selection_x_train = selection.transform(x_train)
print(selection_x_train.shape)

selection_model = GridSearchCV(model, parameters)
selection_model.fit(selection_x_train, y_train)

selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)

score = r2_score(y_test, y_predict)
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.025, selection_x_train.shape[1], score*100))
# Thresh=0.045, n=8, R2:35.77%

# -------------------------------------------------------
selection_model = selection_model.best_estimator_
selection_model.fit(selection_x_train, y_train)
selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)

score = r2_score(y_test, y_predict)
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.025, selection_x_train.shape[1], score*100))
# Thresh=0.045, n=8, R2:35.77%

