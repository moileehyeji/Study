import numpy as np 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

x, y = load_boston(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

model = XGBRegressor(n_jobs = 8)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('r2 : ', score)
# r2 :  0.9221188601856797


# ----------------------------------------------------------------중요도 오름차순
thresholds = np.sort(model.feature_importances_)    #오름차순
print(thresholds)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]
print(np.sum(thresholds))   #1.0



# ----------------------------------------------------------------SelectFromModel
# 중요도 가중치를 기반으로 기능을 선택하기위한 메타 트랜스포머.
# Xgbooster, LGBM, RandomForest등 feature_importances_기능을 쓰는 모델이면 사용 가능
# prefit: fit선행 여부
# False(기본값)이면 transform전에 fit이 선행되어야 하고, 모델을 훈련 fit한 다음 transform기능 선택을 수행
# True이면 transform전에 fit을 안해도 된다, , transform직접 호출
# coef_?
#-------------------------------------------------------prefit=True
# for thresh in thresholds:
#     selection = SelectFromModel(model, 
#                                 threshold = thresh, #Feature 선택에 사용할 임계 값
#                                 prefit=True         #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
#                                 )
#-------------------------------------------------------prefit=False
for thresh in thresholds:
    selection = SelectFromModel(model, 
                                threshold = thresh, #Feature 선택에 사용할 임계 값
                                prefit=False         #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                                ).fit(x_train, y_train)


    # x_train을 선택한 Feature로 줄입니다.
    selection_x_train = selection.transform(x_train)
    print(selection_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, selection_x_train.shape[1], score*100))


# (404, 13)
# Thresh=0.001, n=13, R2:92.21%
# (404, 12)
# Thresh=0.004, n=12, R2:92.16%
# (404, 11)
# Thresh=0.012, n=11, R2:92.03%
# (404, 10)
# Thresh=0.012, n=10, R2:92.19%
# (404, 9)
# Thresh=0.014, n=9, R2:93.08%
# (404, 8)
# Thresh=0.015, n=8, R2:92.37%
# (404, 7)
# Thresh=0.018, n=7, R2:91.48%
# (404, 6)
# Thresh=0.030, n=6, R2:92.71%
# (404, 5)
# Thresh=0.042, n=5, R2:91.74%
# (404, 4)
# Thresh=0.052, n=4, R2:92.11%
# (404, 3)
# Thresh=0.069, n=3, R2:92.52%
# (404, 2)
# Thresh=0.301, n=2, R2:69.41%
# (404, 1)
# Thresh=0.428, n=1, R2:44.98%                    


#------------------------------------------coef
print(model.coef_)
print(model.intercept_)
# AttributeError: Coefficients are not defined for Booster type None
# 부스터 유형 없음에 대한 계수가 정의되지 않았습니다.