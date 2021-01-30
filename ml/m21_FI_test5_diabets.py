# m20_plot_feature_importances2_cancer 복사

# 실습
# 피쳐임포턴스가 0안 컬럼들을 제거하여 데이터셋 재구성 후
# DecisionTreeClassifier로 모델을 돌려서 acc확인
# 오름

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris,load_breast_cancer, load_boston, load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
model1 = DecisionTreeRegressor(max_depth=4)

# 3. 훈련
model1.fit(x_train, y_train)

# 4. 평가, 예측
r2 = model1.score(x_test, y_test)

# ==========================================================================feature_importances_
# feature_importances_ : 컬럼의 중요도 표시
# 해당모델의 중요도를 표시한것으로 모델마다 다르다.
print('컬럼 정리 전 FI  : ', model1.feature_importances_)      
print('컬럼 정리 전 r2 : ', r2)  
print(dataset.data.shape)                


# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
    plt.yticks(np.arange(n_feature), dataset.feature_names)                         # y축 
    plt.title('boston')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model1)
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
        if (original[i] > (original.max() * 0.25)) :          # 중요도가 하위 25프로보다 큰 컬럼만 append
            data_new.append(dataset.data[:,i])
            feature_names.append(dataset.feature_names[i])


data_new = np.array(data_new)
data_new = np.transpose(data_new)

dataset.data = data_new
dataset.feature_names = feature_names

# 전처리
x2_train,x2_test,y2_train,y2_test = train_test_split(data_new,dataset.target, train_size = 0.8, random_state = 33)

#2. 모델
model2 = DecisionTreeRegressor(max_depth = 4)

#3. 훈련
model2.fit(x2_train, y2_train)

#4. 평가 예측
r2 = model2.score(x2_test,y2_test)

print('컬럼 정리 후 FI  : ', model2.feature_importances_)
print('컬럼 정리 후 r2 : ', r2)
print(data_new.shape)                


####### dataset -> new_data 로 변경, feature_name 부분을 feature 리스트로 변경
plot_feature_importances_dataset(model2)
plt.show()

'''
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

'''
