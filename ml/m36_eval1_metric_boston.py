# eval_set
# evals_result

''' 
# eval_metric

eval_metric: 설정 한 objective기본 설정 값이 지정되어 있습니다.
rmse : 제곱 평균 제곱근 오차
mae : 절대 오류를 의미
logloss : 음의 로그 우도
오류 : 이진 분류 오류율 (0.5 임계 값)
merror : 다중 클래스 분류 오류율
mlogloss : 다중 클래스 logloss
auc : 곡선 아래 영역 
'''


# m35_eval1_boston 복사

# eval_metrice다중 가능여부 확인


from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

# 1. 데이터
# x, y = load_boston(return_X_y=True)
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=66)

# 2. 모델구성
model = XGBRegressor(n_estimators = 100, learning_rate = 0.01, n_jobs = 8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'logloss', 'mae','mape'], eval_set=[(x_train, y_train),(x_test, y_test)])      # loss의 history반환

# 4. 평가, 예측
aaa = model.score(x_test, y_test)
print('aaa : ',aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)       # 인자 순서 주의(실제값, 예측값)
print('r2  : ', r2)


# ===================================================================================evals_result
# SearchCV를 모델의 경우 : model.best_estimator_.evals_result()
result = model.evals_result()       # 훈련셋'validation_0', 검증셋'validation_1'
# print('result : ', result)          # {'validation_0': OrderedDict([('rmse'---'validation_1': OrderedDict([('rmse',      



'''
< 회귀의 eval_metric >
[eval_metric='rmse']    rmse :  0.9329663244922279
[r2_score]              r2   :  0.9329663244922279
'''