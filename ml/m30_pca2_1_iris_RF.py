# 컬럼의 양이 방대해질경우 속도, 자원낭비가 심해짐
# feature_importance로 컬럼 줄이기
# pca로 압축
# dense로도 돌아가는 mnist구현

# pca : 특성을 추출하고 압축했을 때 특성 몇개까지가 좋을지 판단할 수 있는 지표
#       데이터 변형이 일어남 --> 전처리같은 의미로 이해
# 데이터를 잘 설명할 수 있는 변수들의 조합을 찾는 것이 목표

# FI는 중요도를 판단 하는 것 -> 자르는 것은 내가 별도로 해

# 랜포로 모델링하시오

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.decomposition import PCA       #분해
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1.데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)       # (150, 4) (150,)

#================================================================PCA : 컬럼 재구성(압축), 차원축소
pca = PCA(n_components=3)
x2 = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, shuffle = True, random_state = 66)


# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)

print('score : ', score)        # score :  1.0




