# 당뇨병 만들어 봐!
# 0.5이상!

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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

# parameters       
parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],  'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]

x, y = load_diabetes(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

# model = XGBRegressor(n_jobs = 8)
model = GridSearchCV(XGBRegressor(n_jobs = 8), parameters)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('최적의 파라미터 : ', model.best_estimator_)
print('r2 : ', score)
# r2 :  0.35207400773121755




model = model.best_estimator_   # 최적의 파라미터로 모델 생성
# ----------------------------------------------------------------중요도 오름차순
thresholds = np.sort(model.feature_importances_)    #오름차순
print(thresholds)
# [0.03880714 0.04463614 0.04701128 0.05405929 0.05463831 0.06191718
#  0.07093667 0.07928223 0.19052564 0.3581862 ]
print(np.sum(thresholds))   #1.0



# ----------------------------------------------------------------SelectFromModel
# 중요도 가중치를 기반으로 기능을 선택하기위한 메타 트랜스포머.
# Xgbooster, LGBM, RandomForest등 feature_importances_기능을 쓰는 모델이면 사용 가능
# prefit: True 인 경우 transform직접 호출 
# coef_?
''' for thresh in thresholds:
    selection = SelectFromModel(model, 
                                threshold = thresh, #Feature 선택에 사용할 임계 값, 중요도가 크거나 같은 기능은 유지되고 나머지는 삭제
                                prefit=True         #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                                )

    # x_train을 선택한 Feature로 줄입니다.
    selection_x_train = selection.transform(x_train)
    print(selection_x_train.shape)  #(353, 10)

    selection_model = GridSearchCV(XGBRegressor(n_jobs = 8), parameters)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, selection_x_train.shape[1], score*100))

# (353, 10)
# Thresh=0.039, n=10, R2:35.21%
# (353, 9)
# Thresh=0.045, n=9, R2:37.77%
# (353, 8)
# Thresh=0.047, n=8, R2:35.77%
# (353, 7)
# Thresh=0.054, n=7, R2:34.71%
# (353, 6)   '''


selection = SelectFromModel(model, 
                            threshold = 0.045, #Feature 선택에 사용할 임계 값, 중요도가 크거나 같은 기능은 유지되고 나머지는 삭제
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
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.045, selection_x_train.shape[1], score*100))
# Thresh=0.025, n=6, R2:90.96%

# -------------------------------------------------------
selection_model = selection_model.best_estimator_
selection_model.fit(selection_x_train, y_train)
selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)

score = r2_score(y_test, y_predict)
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.025, selection_x_train.shape[1], score*100))
# Thresh=0.025, n=6, R2:90.96%

