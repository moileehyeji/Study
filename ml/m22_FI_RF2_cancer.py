# m21_FI_test1_iris

# 실습
# 피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후
# RandomForestClassifier 모델을 돌려서 acc확인
# 동일

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_breast_cancer()

# ===============================================================피쳐임포턴스가 0안 컬럼들을 제거

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

# print(df.shape) #(569, 30)

df = df.iloc[:,[0,1,2,3,5,6,7,10,12,13,20,21,22,23,24,26,27,28]]


dataset.data = df.to_numpy()
# ===============================================================

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

# ==========================================================================feature_importances_
# feature_importances_ : 컬럼의 중요도 표시
# 해당모델의 중요도를 표시한것으로 모델마다 다르다.
print(model.feature_importances_)      

print('acc : ', acc)                  


# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
    # plt.yticks(np.arange(n_feature), dataset.feature_names)       #ValueError: The number of FixedLocator locations (6), usually from a call to set_ticks, does not match the number of ticklabels (30).
    plt.title('cancer')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 전:
[0.01863554 0.01286987 0.03963879 0.06865359 0.00905908 0.01084203
 0.05662244 0.12657942 0.00546327 0.00605096 0.01259337 0.00484001
 0.01739696 0.03463229 0.00427459 0.00575904 0.00785478 0.00554918
 0.00343431 0.00410424 0.07984624 0.0226568  0.1253662  0.12865351
 0.01190804 0.00923761 0.02268093 0.12725114 0.01042869 0.00711707]
acc :  0.9649122807017544


피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후: [0,1,2,3,5,6,7,10,12,13,20,21,22,23,24,26,27,28]
[0.05245507 0.01702452 0.02819842 0.04010401 0.01028997 0.02931114
 0.18381968 0.00847633 0.01177515 0.01720183 0.1094365  0.02940748
 0.15751929 0.13215814 0.02584653 0.02217959 0.11103202 0.01376434]
acc :  0.9649122807017544

'''
