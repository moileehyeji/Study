# m04_iris 복사
# 실습 : 결과 비교
# 회귀모델

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델 import
# from sklearn.svm import  LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# 훈련 loop
scalers = np.array([MinMaxScaler(), StandardScaler()])
models = np.array([LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()])

for j in scalers:

    # x 전처리
    scaler = j
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print('==========================================', scaler)

    for i in models:
        print(i,'   :')

        #2. 모델구성
        model = i

        #3. 훈련
        model.fit(x_train, y_train)

        #4.평가, 예측
        y_pred = model.predict(x_test)
        print('예측값 : ', y_pred[:5])
        print('실제값 : ', y_test[:5])

        result = model.score(x_test, y_test)
        print('model.score     :', result)

        # accuracy_score = accuracy_score(y_test, y_pred)  
        # print('accuracy_score  :', accuracy_score)      #TypeError: 'numpy.float64' object is not callable
        print('r2_score  :', r2_score(y_test, y_pred))

        
        print('\n')   



'''
Tensorflow            :
Conv1D모델 r2 :  0.6436679568820876

========================================== MinMaxScaler()       ***LinearRegression
LinearRegression()    :
예측값 :  [178.35474798 100.7430272  117.89294739 122.05798149 195.77304005]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.4439794579226519
r2_score    : 0.4439794579226519

KNeighborsRegressor()    :
예측값 :  [130.4 117.2 161.4 135.6 186.8]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.3828413917182203
r2_score    : 0.3828413917182203

DecisionTreeRegressor()    :
예측값 :  [138. 144.  78. 111. 181.]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.041263920892073425
r2_score    : 0.041263920892073425

RandomForestRegressor()    :
예측값 :  [151.71 121.01 152.8  127.7  210.08]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.34994855017328286
r2_score    : 0.34994855017328286


========================================== StandardScaler()         ***LinearRegression
LinearRegression()    :
예측값 :  [178.35474798 100.7430272  117.89294739 122.05798149 195.77304005]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.4439794579226519
r2_score    : 0.4439794579226519


KNeighborsRegressor()    :
예측값 :  [113.  117.2 161.4 135.6 186.8]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.38241949480812454
r2_score    : 0.38241949480812454


DecisionTreeRegressor()    :
예측값 :  [ 48. 144.  78. 111. 185.]
실제값 :  [107.  69.  69.  51. 292.]
model.score : -0.16533715127609505
r2_score    : -0.16533715127609505


RandomForestRegressor()    :
예측값 :  [165.09 124.11 143.92 124.5  211.46]
실제값 :  [107.  69.  69.  51. 292.]
model.score : 0.380655353870038
r2_score    : 0.380655353870038


'''