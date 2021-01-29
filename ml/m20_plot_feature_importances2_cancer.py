
# m19_feature_importances 복사

# 모델비교

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# 2. 모델구성
model = DecisionTreeClassifier(max_depth=4)

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
    plt.yticks(np.arange(n_feature), dataset.feature_names)
    plt.title('cancer')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()

'''
[0.         0.         0.         0.         0.         0.00677572
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.00677572 0.         0.
 0.         0.         0.01008994 0.05612587 0.78000877 0.
 0.02293065 0.         0.         0.11729332 0.         0.        ]
acc :  0.9385964912280702
'''
