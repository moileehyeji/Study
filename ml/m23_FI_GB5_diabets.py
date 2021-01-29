# m22_FI_RF1_iris

# 실습
# 피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후
# GradientBoostingRegressor 모델을 돌려서 acc확인
# 높아짐


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris,load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_diabetes()

# ===============================================================피쳐임포턴스가 0안 컬럼들을 제거

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

print(df.shape) #(442, 10)

df = df.iloc[:, [2,3,8]]


dataset.data = df.to_numpy()
# ===============================================================

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model = DecisionTreeClassifier(max_depth=4)
model = GradientBoostingRegressor()

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
    plt.title('diabets')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 전:
[0.07567993 0.01288839 0.27331041 0.08007598 0.03508916 0.06808623
 0.04079109 0.01216416 0.35098086 0.05093379]
acc :  0.36234241174478576


피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후: [2,3,8]
[0.34339897 0.14196811 0.51463292]
acc :  0.3839837321750964

'''
