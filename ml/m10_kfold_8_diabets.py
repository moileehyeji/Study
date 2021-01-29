# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test : validation(1/n)

# 모델별 비교


import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

kfold = KFold(n_splits=5, shuffle=True)

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

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('scores : ', scores)

        '''
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



'''
Tensorflow            :
Conv1D모델 r2 :  0.6436679568820876

======================================================================================1. 기본

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

======================================================================================2. 기본 + Kfold
========================================== MinMaxScaler()
LinearRegression()    :
scores :  [0.58040861 0.28138356 0.41295497 0.580077   0.59934553]
KNeighborsRegressor()    :
scores :  [0.33183558 0.39754046 0.47945989 0.4341282  0.28731895]  ****
DecisionTreeRegressor()    :
scores :  [ 0.05573266  0.14444795 -0.30337258  0.23128874 -0.43778456]
RandomForestRegressor()    :
scores :  [0.47655949 0.30085236 0.4190627  0.24667892 0.585158  ]
========================================== StandardScaler()
LinearRegression()    :
scores :  [0.40103677 0.4811519  0.63557331 0.52349067 0.42327878]  ****
KNeighborsRegressor()    :
scores :  [0.36741946 0.29298477 0.36023292 0.34625086 0.56537959]
DecisionTreeRegressor()    :
scores :  [-0.33626719 -0.36756464  0.06462986  0.22379442 -0.09792741]
RandomForestRegressor()    :
scores :  [0.481171   0.39516973 0.30909884 0.42642669 0.56882898]


'''