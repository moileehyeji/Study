# m20_plot_feature_importances2_cancer 복사

# 실습
# 피쳐임포턴스가 0안 컬럼들을 제거하여 데이터셋 재구성 후
# DecisionTreeClassifier로 모델을 돌려서 acc확인
# 동일

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris,load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_boston()

# ===============================================================피쳐임포턴스가 0안 컬럼들을 제거

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

print(df.shape) #(506, 13)

df = df.iloc[:,[0,4,5,6,7,10,11,12]]


dataset.data = df.to_numpy()
# ===============================================================

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
model = DecisionTreeRegressor(max_depth=4)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
r2 = model.score(x_test, y_test)

# ==========================================================================feature_importances_
# feature_importances_ : 컬럼의 중요도 표시
# 해당모델의 중요도를 표시한것으로 모델마다 다르다.
print(model.feature_importances_)      

print('r2 : ', r2)                  


# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')    # barh : 가로 막대 그래프 , align : 정렬
    # plt.yticks(np.arange(n_feature), dataset.feature_names)       #ValueError: The number of FixedLocator locations (6), usually from a call to set_ticks, does not match the number of ticklabels (30).
    plt.title('boston')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
1. DecisionTreeClassifier 모델 : 
피쳐임포턴스가 0안 컬럼들을 제거하여 데이터셋 재구성 전:
[0.04064049 0.         0.         0.         0.02341569 0.61396724
 0.00869336 0.09447305 0.         0.         0.01858958 0.00343964
 0.19678095]
r2 :  0.8159350178696477


피쳐임포턴스가 0안 컬럼들을 제거하여 데이터셋 재구성 후: [0,4,5,6,7,10,11,12]
[0.02800473 0.02341569 0.61396724 0.00580415 0.09723421 0.01858958
 0.01620345 0.19678095]
r2 :  0.8159350178696476

'''
