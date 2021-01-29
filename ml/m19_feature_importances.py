

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
dateset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dateset.data, dateset.target, test_size=0.2, random_state=44)

# 2. 모델구성
model = DecisionTreeClassifier(max_depth=4)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

# ==========================================================================feature_importances_
# feature_importances_ : 컬럼의 중요도 표시
# 해당모델의 중요도를 표시한것으로 모델마다 다르다.
print(model.feature_importances_)       # [0.         0.         0.43843499 0.56156501]

print('acc : ', acc)                    # acc :  0.9333333333333333
