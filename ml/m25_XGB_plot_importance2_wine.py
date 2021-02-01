# XGBBoost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_wine()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model1 = DecisionTreeClassifier(max_depth=4)
# model1 = RandomForestClassifier()
# model1 = GradientBoostingClassifier()
model1 = XGBClassifier(n_jobs=-1, eval_metric='mlogloss') 

# 3. 훈련
model1.fit(x_train, y_train)

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
#     plt.title('cancer')
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
        if (original[i] > (original.max() * 0.25)) :         # 
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
model2 = XGBClassifier(n_jobs=-1, eval_metric='mlogloss') 


#3. 훈련
model2.fit(x2_train, y2_train)

#4. 평가 예측
acc = model2.score(x2_test,y2_test)

print('컬럼 정리 후 FI  : ', model2.feature_importances_)
print('컬럼 정리 후 acc : ', acc)
print(data_new.shape)                


####### dataset -> new_data 로 변경, feature_name 부분을 feature 리스트로 변경
# plot_feature_importances_dataset(model2)
plot_importance(model2)
plt.show()

'''
< feature_importances_ 낮은 컬럼 제거 전후 모델별 비교 >

1. DecisionTreeClassifier 모델 : 
컬럼 정리 전 FI  :  [0.         0.01723824 0.         0.         0.         0.
                    0.15955687 0.         0.         0.         0.07373053 0.32933594
                    0.42013842]
컬럼 정리 전 acc :  0.9166666666666666
(178, 13)
컬럼 정리 후 FI  :  [0.         0.10338944 0.09679353 0.33148253 0.46833451]
컬럼 정리 후 acc :  0.75
(178, 5)

2. RandomFrest모델 :
컬럼 정리 전 FI  :  [0.1576577  0.04002824 0.01453107 0.02980359 0.02180664 0.07416311
                    0.16893505 0.01070017 0.02205451 0.10148678 0.07265435 0.12683951
                    0.15933926]
컬럼 정리 전 acc :  0.9722222222222222
(178, 13)
컬럼 정리 후 FI  :  [0.12687771 0.04446649 0.15679109 0.17557077 0.08903643 0.13922835
                    0.26802917]
컬럼 정리 후 acc :  0.9722222222222222
(178, 7)

3. GradientBoostingClassifier 모델 :
컬럼 정리 전 FI  :  [5.92822071e-02 3.44182025e-02 1.46276285e-02 1.42170589e-03
                    7.96221009e-03 1.34718855e-07 1.93951345e-01 2.20739191e-03
                    1.60647070e-03 2.22876237e-01 2.11188089e-02 1.48254150e-01
                    2.92273508e-01]
컬럼 정리 전 acc :  0.9166666666666666
(178, 13)
컬럼 정리 후 FI  :  [0.07469242 0.32769544 0.2495111  0.34810104]
컬럼 정리 후 acc :  0.9444444444444444
(178, 4)

4. XGB 모델
컬럼 정리 전 FI  :  [0.06830431 0.04395564 0.00895185 0.         0.01537293 0.00633501
                    0.07699447 0.00459099 0.00464443 0.08973485 0.01806366 0.5588503
                    0.10420163]
컬럼 정리 전 acc :  0.9444444444444444
(178, 13)
컬럼 정리 후 FI  :  [0.02003486 0.08083776 0.00569393 0.02126298 0.01801993 0.10462575
                    0.00455905 0.00949741 0.16046508 0.02633205 0.3872465  0.16142473]
컬럼 정리 후 acc :  1.0
(178, 12)
'''