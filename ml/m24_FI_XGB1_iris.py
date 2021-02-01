# m23_FI_GB1_iris copy

# XGBooster
# CMD : pip install xgboost

# warning : 
'''
model2 = XGBClassifier(n_jobs=-1, use_label_encoder=False)          # use_label_encoder=False

model2.fit(x2_train, y2_train, eval_metric='logloss')               # eval_metric='logloss'
'''



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from time import time

import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()

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
print('n_jobs별 걸린시간 : ', n_jobss)   #n_jobs별 걸린시간 :  [0.07152987 0.03889608 0.03590417 *0.03191471*]   


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

def plot_feature_importances_dataset(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
    plt.yticks(np.arange(n_feature), dataset.feature_names)                         # y축 
    plt.title('iris')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model1)
# plt.show()

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
model2 = XGBClassifier(n_jobs=-1, use_label_encoder=False)           # 코어 모두 사용

#3. 훈련
model2.fit(x2_train, y2_train, eval_metric='logloss')

#4. 평가 예측
acc = model2.score(x2_test,y2_test)

print('컬럼 정리 후 FI  : ', model2.feature_importances_)
print('컬럼 정리 후 acc : ', acc)
print(data_new.shape)                


####### dataset -> new_data 로 변경, feature_name 부분을 feature 리스트로 변경
plot_feature_importances_dataset(model2)
# plt.show()

'''
< feature_importances_ 낮은 컬럼 제거 전후 모델별 비교 >

1. DecisionTreeClassifier 모델 : 
컬럼 정리 전 FI  :  [0.         0.00787229 0.96203388 0.03009382]
컬럼 정리 전 acc :  0.9333333333333333
(150, 4)
컬럼 정리 후 FI  :  [0.01692047 0.57459774 0.40848179]
컬럼 정리 후 acc :  0.8666666666666667

2. RandomForestClassifier 모델 :
컬럼 정리 전 FI  :  [0.08873614 0.02329671 0.50154311 0.38642404]
컬럼 정리 전 acc :  0.9666666666666667
(150, 4)
컬럼 정리 후 FI  :  [0.47917544 0.52082456]
컬럼 정리 후 acc :  0.9
(150, 2)

3. GradientBoostingClassifier 모델 :
컬럼 정리 전 FI  :  [0.00673725 0.01252317 0.66360937 0.31713021]
컬럼 정리 전 acc :  0.9666666666666667
(150, 4)
컬럼 정리 후 FI  :  [0.26055134 0.73944866]
컬럼 정리 후 acc :  0.9
(150, 2)

4. XGB 모델 :
컬럼 정리 전 FI  :  [0.02323038 0.01225644 0.8361378  0.12837538]
컬럼 정리 전 acc :  0.9666666666666667
(150, 4)
컬럼 정리 후 FI  :  [1.]
컬럼 정리 후 acc :  0.9333333333333333
(150, 1)

'''
