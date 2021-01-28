# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test : validation(1/n)

# 모델별 비교

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score

# 모델 import
# from sklearn.svm import  LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
dataset = load_boston()
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
CNN모델 r2 :  0.9462232137123261

======================================================================================1. 기본

========================================== MinMaxScaler()       ***RandomForestRegressor
LinearRegression()    :
예측값 :  [11.88108826 20.96545415 16.79574055 32.06633245 23.12186109]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.776128674514256
r2_score    : 0.776128674514256

KNeighborsRegressor()    :
예측값 :  [10.98 14.42 15.4  36.24 26.46]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.6310388450317691
r2_score    : 0.6310388450317691

DecisionTreeRegressor()    :
예측값 :  [ 8.4 21.7 15.4 28.7 18.7]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.46963790891834467
r2_score    : 0.46963790891834467

RandomForestRegressor()    :
예측값 :  [11.205 21.855 14.53  33.033 24.807]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.8638905443142602
r2_score    : 0.8638905443142602


========================================== StandardScaler()     ***RandomForestRegressor         
LinearRegression()   :
예측값 :  [11.88108826 20.96545415 16.79574055 32.06633245 23.12186109]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.7761286745142559
r2_score    : 0.7761286745142559

KNeighborsRegressor()   :
예측값 :  [ 9.68 15.2  17.64 34.84 25.84]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.7168416000301296
r2_score    : 0.7168416000301296

DecisionTreeRegressor()   :
예측값 :  [ 8.3 23.2 15.2 23.6 18.3]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.5753889097606584
r2_score    : 0.5753889097606584

RandomForestRegressor()   :
예측값 :  [10.91  21.596 14.155 32.646 24.384]
실제값 :  [11.8 16.1 19.1 33.2 26.4]
model.score : 0.86262734908277
r2_score    : 0.86262734908277

======================================================================================2. 기본 + Kfold
========================================== MinMaxScaler()
LinearRegression()    :
scores :  [0.59036284 0.68304044 0.77162772 0.71619525 0.7182398 ]
KNeighborsRegressor()    :
scores :  [0.74709072 0.6631525  0.74978626 0.77809854 0.76927996]
DecisionTreeRegressor()    :
scores :  [0.72142051 0.60640555 0.84935089 0.71444902 0.51899265]
RandomForestRegressor()    :
scores :  [0.82695225 0.86258398 0.85423892 0.88499517 0.83480057]  ****
========================================== StandardScaler()
LinearRegression()    :
scores :  [0.79004358 0.67819186 0.69381002 0.53566393 0.67506366]
KNeighborsRegressor()    :
scores :  [0.81081927 0.63601843 0.59305722 0.85542567 0.7986303 ]
DecisionTreeRegressor()    :
scores :  [0.64083017 0.79743321 0.73009507 0.85498266 0.79192779]
RandomForestRegressor()    :
scores :  [0.88502276 0.87879314 0.84688783 0.86847138 0.89204626]  ****
'''