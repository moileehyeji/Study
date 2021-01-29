# m22_FI_RF1_iris

# 실습
# 피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후
# GradientBoostingClassifier 모델을 돌려서 acc확인
# 동일


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_wine()

# ===============================================================피쳐임포턴스가 0안 컬럼들을 제거

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

print(df.shape) #(178, 13)

df = df.iloc[:,[0,1,2,6,9,10,11,12]]


dataset.data = df.to_numpy()
# ===============================================================

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model = DecisionTreeClassifier(max_depth=4)
model = GradientBoostingClassifier()

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
    plt.title('wine')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 전:
[5.98886804e-02 3.37074904e-02 1.49593886e-02 2.12725709e-03
 7.96488236e-03 4.10863843e-08 2.12146669e-01 2.41005244e-03
 1.30773916e-03 2.29415444e-01 2.03729315e-02 1.23290608e-01
 2.92408817e-01]
acc :  0.9166666666666666


피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후: [0,1,2,6,9,10,11,12]
[0.05702345 0.03682404 0.01528824 0.19427713 0.22277808 0.02131644
 0.15283127 0.29966135]
acc :  0.9166666666666666

'''
