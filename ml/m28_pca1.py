# 컬럼의 양이 방대해질경우 속도, 자원낭비가 심해짐
# feature_importance로 컬럼 줄이기
# pca로 압축
# dense로도 돌아가는 mnist구현

# pca : 특성을 추출하고 압축했을 때 특성 몇개까지가 좋을지 판단할 수 있는 지표
#       데이터 변형이 일어남 --> 전처리같은 의미로 이해
# 데이터를 잘 설명할 수 있는 변수들의 조합을 찾는 것이 목표

# FI는 중요도를 판단 하는 것 -> 자르는 것은 내가 별도로 해


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA       #분해

dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)       # (442, 10) (442,)

#================================================================PCA : 컬럼 재구성(압축)
'''
pca = PCA(n_components=7)
x2 = pca.fit_transform(x)       # fit과 transform 합친것

# print(x2.shape)               # (442, 7)

# 컬럼내역
# explained_variance_ratio_ : 각각의 주성분 벡터가 이루는 축에 투영(projection)한 결과의 분산의 비율
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)                  # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]


# 위의 결과의 의미는 원 데이터셋 분산의 40.2%가 첫 번째 주성분 축에 놓여 있고, 14.9%가 두 번째 주성분 축에 놓여 있다는 것을 말한다. 그리고 세 번째 주성분 축에는 12.0% 정도 양의 정보가 들어 있다는 것을 알 수 있다.


print(sum(pca_EVR))             #            

# 통상적으로 95%까지는 성능이 비슷하다.
# 7개 : 0.9479436357350414
# 8개 : 0.9913119559917797
# 9개 : 0.9991439470098977
# 10개: 1.0

'''

pca = PCA()
pca.fit(x)

# 적절한 차원 수 선택하기
# 'Explained Variance Ratio'를 이용해 축소할 차원의 수를 선택할 수 있다. 예를들어, 누적된 분산의 비율이 99%가 되는 주성분 축, 차원을 선택하는 것과 같은 방법이다.
# cumsum : 각 원소들의 누적 합을 표시함. 각 row와 column의 구분은 없어지고, 순서대로 sum을 함.

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)      #  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]
                # n_components=          1          2          3         4           5          6           7         8          9          10

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95   :', cumsum >= 0.95)
print('선택할 차원의 수  : ', d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
