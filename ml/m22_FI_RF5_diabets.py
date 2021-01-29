# m21_FI_test1_iris

# 실습
# 피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후
# RandomForestRegressor 모델을 돌려서 acc확인
# 떨어짐


def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:       # 중요도 리스트
        temp.append(i)
    temp.sort()                         # 오름차순
    temp=temp[:number]                  # 지정한 number만큼 대입(중요도 작은애들)
    result = []
    for j in temp:                      # 해당 인덱스 찾아서 ...
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

df = df.iloc[:,[0,2,3,4,5,6,7,9]]


dataset.data = df.to_numpy()
# ===============================================================

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestRegressor()

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
    plt.title('boston')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 전:
[0.07463619 0.00943346 0.26587718 0.07946509 0.05011016 0.06254609
 0.05443621 0.02257845 0.31416131 0.06675587]
acc :  0.41282530608895496


피쳐임포턴스가 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후: [0,2,3,4,5,6,7,9]
[0.07734562 0.40033761 0.11130273 0.0693592  0.06776036 0.09151352 0.07643834 0.10594261]
acc :  0.3618197905307071

'''
