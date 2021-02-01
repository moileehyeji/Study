# 실습
# pca통해서 0.95이상인거 몇개?
# pca 배운거 다 집어넣고 확인

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()   # _는 y_train, y_test 사용XXX

x = np.append(x_train, x_test, axis=0)

print(x.shape)      #(70000, 28, 28)

# ValueError: Found array with dim 3. Estimator expected <= 2.
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])


pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum >= 1.0) + 1
print('cumsum >= 1.0  :', cumsum >= 1.0)
print('선택할 차원의 수 :', d)              # 0.95 -> 선택할 차원의 수 : 154,    1.0-> 선택할 차원의 수 : 713



