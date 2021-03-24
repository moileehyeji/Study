# https://dsbook.tistory.com/192
# polynomial 사이킷런 공식문서
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py

# polynomial이란?

# polynomial : 다항(식)의

# sklearn.preprocessing.PolynomialFeatures( degree = 2 , * , interaction_only = False , include_bias = True , order = 'C')
# * degree int, default = 2
#     다항식 기능의 정도입니다.
# * interaction_only bool, 기본값 = False
#     true인 경우에만 상호 작용 기능은 생산 : 최대의 제품은 기능이 degree 서로 다른 입력 기능 
#     true : [1, a, b, ab]
# * include_bias bool, 기본값 = True
#     True (기본값)이면 모든 다항식 거듭 제곱이 0 인 특성 인 편향 열을 포함합니다 (즉, 1로 구성된 열-선형 모델에서 절편 항의 역할을 함).
# * order { 'C', 'F'}, 기본값 = 'C'
#     조밀 한 경우의 출력 배열 순서입니다. 'F'차수는 계산 속도가 더 빠르지 만 후속 추정기를 느리게 할 수 있습니다.



# 다항식 및 상호 작용 기능을 생성합니다.
# 차수가 지정된 차수보다 작거나 같은 특성의 모든 다항식 조합으로 구성된 새 특성 행렬을 생성합니다. 
# 예) 입력 샘플이 2 차원이고 [a, b] 형식인 경우 2차 다항식 특징은 [1, a, b, a ^ 2, ab, b ^ 2]


# 편향 - 분산 trade off
# 선형회귀의 경우에, 데이터를 충분히 표현하기 못하는 경우가 발생할 수 있다. (편향이 매우 큼)
# 반대로 다항회귀에서 차수가 너무 큰 경우 변동성이 커지고, 이를 고분산성을 가진다고 이야기한다.

# 편향(bias)와 분산(variance)는 한쪽은 성능을 좋게하고, 다른 한쪽의 성능이 떨어지는 trade-off의 관계에 있다.
# 쉽게 말해 편향을 줄이면 분산이 늘어나고, 분산을 줄이면 편향이 늘어난다고 보면 된다.

# 따라서 둘의 성능을 적절하게 맞춰 전체 오류가 낮아지는 지점을 Goldilocks(골디락스)지점이라고 한다.
# 이 지점을 찾는것이 매우 중요라다.

# 그렇다면 직선보다 복잡한 회귀선이 성능이 더 좋다? 아니다!!
# 과적합문제때문에 예측값의 성능이 저하되는 현상이 발생한다. 
# 그래서 다항식의 차수를 결정하는 것이 매우 중요하다.
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
# 다항피처의 차수를 변화시키면서 예측의 정확도를 비교하는 내용

#다항식의 차수가 지나치게 높으면 오히려 왜곡
# 차수가 너무 작으면 데이터를 충분히 표현하지 못함

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
print('일차 단항식 계수 피처:\n', X)
# array([[0, 1],
#        [2, 3],
#        [4, 5]])


poly = PolynomialFeatures(degree=2)
X1 = poly.fit_transform(X)
print('변환된 2차 다항식 계수 피처:\n', X1)


# array([[ 1.,  0.,  1.,  0.,  0.,  1.],
#        [ 1.,  2.,  3.,  4.,  6.,  9.],
#        [ 1.,  4.,  5., 16., 20., 25.]])


poly = PolynomialFeatures(interaction_only=True)
X2 = poly.fit_transform(X)
print('interaction_only=True:\n', X2)
# array([[ 1.,  0.,  1.,  0.],
#        [ 1.,  2.,  3.,  6.],
#        [ 1.,  4.,  5., 20.]])


#==================================원하는 다항식을 만드는 경우 함수정의
def polynomial_func(X): 
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3
    return y

y = polynomial_func(X)
print('사용자정의 3차 다항식 결정값 :\n ', y)
#   [  5 125 557]


#==================================선형회귀 이용하여 회귀 계수찾기
poly = PolynomialFeatures(degree=3)
X3 = poly.fit_transform(X)
print('변환된 3차 다항식 계수 피처:\n', X3)
#  [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X3, y)

print('Polynomial 회귀 계수: \n', np.round(model.coef_, 2))
print('Polynomial 회귀 모양: \n', model.coef_.shape)
# Polynomial 회귀 계수: 
#  [ 0.    0.47  0.47  0.57  1.04  1.51 -0.3   0.27  1.31  2.81]
# Polynomial 회귀 모양:
#  (10,)


#====================================피처변환과 선형회귀의 적용 한번에: Pipeline
from sklearn.pipeline import Pipeline

model = Pipeline([ ('poly', PolynomialFeatures(degree=3)), 
                   ('linear', LinearRegression()) ])

X = np.arange(4).reshape(2,2)
y = polynomial_func(X)
model.fit(X, y)

print('Polynomial 회귀 계수 : \n', np.round(model.named_steps['linear'].coef_,2))
#  [0.   0.18 0.18 0.36 0.54 0.72 0.72 1.08 1.62 2.34]




#====================================차수에 따른 예측 정확도 비교하기(피처차수 1,4,15 예측결과 비교)
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score

# 데이터 설정
#임의의 값으로 구성된 x값에 대해 코사인 변환 값을 반환
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

#x눈 0-1 30개의 임의의 값을 순서대로 샘플링한 데이터
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))

#y값은 코사인 기반의 true_fun에서 약간의 노이즈 변동값을 더한 값입니다.
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize = (14,5))
degrees = [1, 4, 15]

for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    # 개별 degree별로 polinomial 변환합니다.
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([ ('poly_features', polynomial_features), 
                          ('linear_regression', linear_regression) ])

    pipeline.fit(X.reshape(-1, 1), y)

    # 교차검증으로 다항회귀를 평가
    scores = cross_val_score(pipeline, X.reshape(-1, 1), y,
                                scoring="neg_mean_squared_error", cv=10)

    # pipeline을 구성하는 세부객체를 첩근하는 name_steps['객체명']을 이용해 회귀계수 추출
    coef = pipeline.named_steps['linear_regression'].coef_
    print('\nDegree{0} 회귀계수는 {1}입니다.'.format(degrees[i], np.round(coef, 2)))
    print('Degree {0} MSE는 {1}입니다.'.format(degrees[i], -1*np.mean(scores)))

    #0부터 1까지 테스트 데이터 세트를 100개로 나눠 예측 수행
    #테스트 데이터 세트에 회귀 예측을 수행하고 예측곡선과 실제 곡선을 그려서 비교
    X_test = np.linspace(0, 1, 100)
    #예측값곡선
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    #실제값곡선
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()

# 다항식의 차수가 지나치게 높으면 오히려 왜곡
# 차수가 너무 작으면 데이터를 충분히 표현하지 못함
