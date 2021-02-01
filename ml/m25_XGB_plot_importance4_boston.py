# XGBBoost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor, plot_importance

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_boston()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model1 = DecisionTreeClassifier(max_depth=4)
# model1 = RandomForestClassifier()
# model1 = GradientBoostingClassifier()
model1 = XGBRegressor(n_jobs=-1,  eval_metric='mlogloss')

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
model2 = XGBRegressor(n_jobs=-1,  eval_metric='mlogloss')


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
컬럼 정리 전 FI  :  [0.02800473 0.00288921 0.         0.         0.02341569 0.61396724
                    0.00580415 0.094345   0.         0.         0.01858958 0.01620345
                    0.19678095]
컬럼 정리 전 r2 :  0.8159350178696477
(506, 13)
컬럼 정리 후 FI  :  [0.04032715 0.         0.00697375 0.63787124 0.01685667 0.09266201
                    0.00493084 0.         0.20037834]
컬럼 정리 후 r2 :  0.6454349025870473
(506, 9)

2. RandomFrest모델 :
컬럼 정리 전 FI  :  [0.03876394 0.00070534 0.00542481 0.00101938 0.02241288 0.40496537
                    0.01397878 0.06982437 0.00271344 0.01521658 0.01949164 0.0104032
                    0.39508027]
컬럼 정리 전 r2  :  0.8904320291461743
(506, 13)
컬럼 정리 후 FI  :  [0.51020904 0.48979096]
컬럼 정리 후 r2  :  0.6879644038030498
(506, 2)

3. GradientBoosting 모델 :
컬럼 정리 전 FI  :  [2.70826500e-02 2.89676686e-04 2.57257454e-03 1.13227503e-03
                    3.38148249e-02 3.79842345e-01 8.68309640e-03 9.79546496e-02
                    7.73299797e-04 1.16855650e-02 3.46767799e-02 5.54985503e-03
                    3.95942408e-01]
컬럼 정리 전 r2  :  0.8947375666731425
(506, 13)
컬럼 정리 후 FI  :  [0.47099083 0.52900917]
컬럼 정리 후 r2  :  0.6837207996598044
(506, 2)

4. XGB 모델
컬럼 정리 전 FI  :  [0.01311134 0.00178977 0.00865051 0.00337766 0.03526587 0.24189197
                    0.00975884 0.06960727 0.01454236 0.03254252 0.04658296 0.00757505
                    0.51530385]
컬럼 정리 전 acc :  0.8902902185916939
(506, 13)
컬럼 정리 후 FI  :  [0.45818415 0.5418159 ]
컬럼 정리 후 acc :  0.6382169816726256
(506, 2)
'''