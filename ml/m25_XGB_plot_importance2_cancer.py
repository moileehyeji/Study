# m24_FI_XGB1_iris복사

# plot_importance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from time import time

import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성

# 타임걸어
# n_jobs = -1,8, 4, 1 속도비교
n_jobs_list=[-1,8,4,1]
n_jobss = []

for i in n_jobs_list : 
    # model1 = DecisionTreeClassifier(max_depth=4)
    # model1 = RandomForestClassifier()
    # model1 = GradientBoostingClassifier()
    model1 = XGBClassifier(n_jobs=i, eval_metric='mlogloss')           # 코어 모두 사용

    start_time = time()

    # 3. 훈련
    model1.fit(x_train, y_train)

    finish_time = time()

    n_jobss.append(finish_time-start_time)

n_jobss = np.array(n_jobss)
print('n_jobs별 걸린시간 : ', n_jobss)   #n_jobs별 걸린시간 :  [0.06605697 0.03390932 0.03490686 0.06996441]


# 4. 평가, 예측
acc = model1.score(x_test, y_test)

# ==========================================================================feature_importances_
# feature_importances_ : 컬럼의 중요도 표시
# 해당모델의 중요도를 표시한것으로 모델마다 다르다.
print('컬럼 정리 전 FI  : ', model1.feature_importances_)      
print('컬럼 정리 전 acc : ', acc)  
print(dataset.data.shape)                


# 시각화
import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model):
#     n_feature = dataset.data.shape[1]
#     plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
#     plt.yticks(np.arange(n_feature), dataset.feature_names)                         # y축 
#     plt.title('iris')
#     plt.xlabel('Feature Importance')
#     plt.ylabel('Feature')
#     plt.ylim(-1, n_feature)

# plot_feature_importances_dataset(model1)

plot_importance(model1)
plt.show()

# ============================================================ 중요도가 0인 컬럼 정리(서영이 m21_FI_test1_iris)
## 0인 컬럼 제거
original = model1.feature_importances_
data_new =[]        # 새로운 데이터형성 dataset --> data_new
feature_names = []  # 컬럼 이름 정의 feature_names


# for문 생성-> 중요도 낮은 컬럼 제거
if np.any(0 == original) == True :                          # 중요도에 0 이 있으면
    for i in range(len(original)):
        if (original[i] > 0.) :                             # 중요도가 0 보다 큰 컬럼만 append
            data_new.append(dataset.data[:,i])
            feature_names.append(dataset.feature_names[i])
else :                                                      # 중요도에 0 이 없으면
    for i in range(len(original)):
        if (original[i] > (original.max() * 0.25)) :          # 
            data_new.append(dataset.data[:,i])
            feature_names.append(dataset.feature_names[i])


data_new = np.array(data_new)
data_new = np.transpose(data_new)

dataset.data = data_new
dataset.feature_names = feature_names

# 전처리
x2_train,x2_test,y2_train,y2_test = train_test_split(data_new,dataset.target, train_size = 0.8, random_state = 33)

#2. 모델
# model2 = DecisionTreeClassifier(max_depth = 4)
# model2 = RandomForestClassifier()
# model2 = GradientBoostingClassifier()
model2 = XGBClassifier(n_jobs=-1, eval_metric='mlogloss')            # 코어 모두 사용

#3. 훈련
model2.fit(x2_train, y2_train)

#4. 평가 예측
acc = model2.score(x2_test,y2_test)

print('컬럼 정리 후 FI  : ', model2.feature_importances_)
print('컬럼 정리 후 acc : ', acc)
print(data_new.shape)                


####### dataset -> new_data 로 변경, feature_name 부분을 feature 리스트로 변경
plot_importance(model2)
plt.show()

'''
< feature_importances_ 낮은 컬럼 제거 전후 모델별 비교 >

1. DecisionTreeClassifier 모델 : 
컬럼 정리 전 FI  :  [0.         0.         0.         0.         0.         0.
                    0.         0.         0.         0.         0.         0.
                    0.         0.00677572 0.         0.         0.         0.
                    0.         0.         0.01008994 0.05612587 0.78000877 0.
                    0.00995429 0.         0.         0.1370454  0.         0.        ]
컬럼 정리 전 acc :  0.9385964912280702
(569, 30)

컬럼 정리 후 FI  :  [0.         0.00111657 0.0413355  0.82018966 0.05925333 0.07810494]
컬럼 정리 후 acc :  0.9298245614035088
(569, 6)

2. RandomFrest모델 :
컬럼 정리 전 FI  :  [0.02604947 0.01725403 0.05962949 0.03780032 0.00825135 0.01528129
                    0.07501805 0.08328476 0.00435496 0.00635749 0.02031432 0.00615687
                    0.00765118 0.04376177 0.00512134 0.0046758  0.01140007 0.00732979
                    0.00327074 0.00427899 0.14537556 0.01476771 0.15375189 0.06802011
                    0.01108553 0.01627374 0.03700081 0.08081024 0.01837657 0.00729575]
컬럼 정리 전 acc :  0.9649122807017544
(569, 30)
컬럼 정리 후 FI  :  [0.06034728 0.04506332 0.20436503 0.02566116 0.15251015 0.16769284
                    0.20731282 0.13704739]
컬럼 정리 후 acc :  0.9210526315789473
(569, 8)

3. GradientBoostingClassifier 모델 :
컬럼 정리 전 FI  :  [1.19839910e-03 8.22968275e-03 7.25101062e-04 5.72326065e-04
                    1.84196435e-05 3.24710265e-03 9.18929641e-04 4.09268304e-01
                    1.44769931e-04 1.42412391e-03 1.46804192e-03 2.33757924e-03
                    2.25967950e-03 7.57893642e-03 3.72306479e-04 1.46312913e-03
                    1.43575260e-03 1.87793837e-03 6.30697081e-04 3.16040513e-03
                    7.16929717e-02 6.28374250e-02 2.92311305e-01 3.51105959e-02
                    1.33121678e-02 3.86315896e-03 1.90406965e-02 5.25071529e-02
                    9.67115000e-04 2.57863493e-05]
컬럼 정리 전 acc :  0.9736842105263158
(569, 30)
컬럼 정리 후 FI  :  [0.17012748 0.82987252]
컬럼 정리 후 acc :  0.8947368421052632
(569, 2)

4. XGB모델 :
컬럼 정리 전 acc :  0.9824561403508771
(569, 30)
컬럼 정리 후 FI  :  [0.00972049 0.01461379 0.         0.03154194 0.01624345 0.00369781
                    0.00820122 0.0486874  0.00092533 0.00058826 0.01139637 0.00248311
                    0.00340085 0.00337634 0.00397922 0.00667666 0.         0.00229014
                    0.00186384 0.01358143 0.16155778 0.01727752 0.49462706 0.0141327
                    0.005512   0.04696959 0.0700702  0.00130578 0.00527969]
컬럼 정리 후 acc :  0.9473684210526315
(569, 29) 

'''
