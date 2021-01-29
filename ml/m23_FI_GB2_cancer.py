# m22_FI_RF1_iris

# 실습
# 피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후
# GradientBoostingClassifier 모델을 돌려서 acc확인
# 떨어짐


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. 데이터
dataset = load_breast_cancer()

# ===============================================================피쳐임포턴스가 0안 컬럼들을 제거

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

print(df.shape) #(569, 30)

df = df.iloc[:,[1,7,20,21,22,23,24,26,27]]


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
    plt.title('cancer')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 전:
[5.57639576e-05 1.34459275e-02 3.29557722e-04 1.24475453e-03
 1.60860686e-03 3.39132193e-03 5.61416814e-04 4.07273548e-01
 1.69683688e-03 2.23803874e-03 2.23364042e-03 1.91558929e-03
 1.77911129e-03 7.23865764e-03 2.52428342e-04 5.01137054e-04
 6.22953660e-04 9.85328885e-04 1.70546070e-03 3.63534717e-03
 6.09884595e-02 5.79220932e-02 2.88558397e-01 5.12325293e-02
 1.33136260e-02 7.39860048e-04 1.93492973e-02 5.42001022e-02
 7.22083592e-04 2.58124349e-04]
acc :  0.9824561403508771


피쳐임포턴스가 0 or 25% 아래 컬럼들을 제거하여 데이터셋 재구성 후: [1,7,20,21,22,23,24,26,27]
[0.01567705 0.41096315 0.08151168 0.06043648 0.27596446 0.05362574
 0.01459816 0.02270076 0.06452252]
acc :  0.9649122807017544

'''
