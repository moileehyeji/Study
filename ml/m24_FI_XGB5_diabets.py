# XGBBoost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model1 = DecisionTreeClassifier(max_depth=4)
# model1 = RandomForestClassifier()
# model1 = GradientBoostingClassifier()
model1 = XGBRegressor(n_jobs=-1, eval_metric='mlogloss') 

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

def plot_feature_importances_dataset(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
    plt.yticks(np.arange(n_feature), dataset.feature_names)                         # y축 
    plt.title('cancer')
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
model2 = XGBRegressor(n_jobs=-1, eval_metric='mlogloss') 


#3. 훈련
model2.fit(x2_train, y2_train)

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
컬럼 정리 전 FI  :  [4.03463610e-02 2.16063823e-04 3.19268150e-01 0.00000000e+00
                    1.83192445e-02 4.99774661e-02 0.00000000e+00 0.00000000e+00
                    5.70591853e-01 1.28086228e-03]
컬럼 정리 전 r2 :  0.22679038813478902
(442, 10)
컬럼 정리 후 FI  :  [0.03171457 0.         0.344041   0.01293265 0.07835212 0.49550822
                    0.03745144]
컬럼 정리 후 r2 :  0.38945127302905436
(442, 7)

2. RandomFrest모델 :
컬럼 정리 전 FI  :  [0.06367395 0.01274027 0.24806163 0.08020497 0.04523876 0.06390351
                    0.05111634 0.02028922 0.34829193 0.06647942]
컬럼 정리 전 r2  :  0.41337016009520533
(442, 10)
컬럼 정리 후 FI  :  [0.49325093 0.50674907]
컬럼 정리 후 r2  :  0.3664797071931054
(442, 2)

4. XGB 모델
컬럼 정리 전 FI  :  [0.0368821  0.03527097 0.15251055 0.05477958 0.04415327 0.06812558
                    0.0651588  0.05049536 0.42164674 0.0709771 ]
컬럼 정리 전 acc :  0.24138193114785134
(442, 10)
컬럼 정리 후 FI  :  [0.38895524 0.61104476]
컬럼 정리 후 acc :  0.13676951114355862
(442, 2)
'''