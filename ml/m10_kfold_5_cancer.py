# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test : validation(1/n)

# 모델별 비교

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     #--> 분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77, shuffle = True)

# n등분 설정
kfold = KFold(n_splits=5, shuffle=True) 

#2. 모델구성

scalers = np.array([MinMaxScaler(), StandardScaler()])
models = np.array([LinearSVC(), SVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()])

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

            scores = cross_val_score(model, x_train, y_train, cv=kfold)     # train : test : validation(x_train 1/5)
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
            print('accuracy_score  :', accuracy_score(y_test, y_pred))

            
            print('\n')  
            ''' 




'''
Tensorflow                 :
Dense(Dropout)모델 acc :  0.9912280440330505

======================================================================================1. 기본
                    <MinMaxScaler>          ***LinearSVC
1. LinearSVC               :  
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9736842105263158
    accuracy_score  : 0.9736842105263158

2. SVC                     :  
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.956140350877193
    accuracy_score  : 0.956140350877193       

3. KNeighborsClassifier    :
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.956140350877193
    accuracy_score  : 0.956140350877193

4. DecisionTreeClassifier  :
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9298245614035088
    accuracy_score  : 0.9298245614035088

5. RandomForestClassifier  :
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9649122807017544
    accuracy_score  : 0.9649122807017544


                     <StandardScaler>       ***LinearSVC, SVC, RandomForestClassifier

1. LinearSVC               :  
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9736842105263158
    accuracy_score  : 0.9736842105263158

2. SVC                     : 
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9736842105263158
    accuracy_score  : 0.9736842105263158   

3. KNeighborsClassifier    : 
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.956140350877193
    accuracy_score  : 0.956140350877193

4. DecisionTreeClassifier  : 
    예측값 :  [1 1 1 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9298245614035088
    accuracy_score  : 0.9298245614035088

5. RandomForestClassifier  : 
    예측값 :  [1 1 0 1 1]
    실제값 :  [1 1 0 0 1]
    model.score     : 0.9736842105263158
    accuracy_score  : 0.9736842105263158


======================================================================================2. 기본 + Kfold

========================================== MinMaxScaler()
LinearSVC()    :
scores :  [0.98901099 1.         0.95604396 0.97802198 0.96703297]  ****
SVC()    :
scores :  [1.         0.97802198 0.95604396 0.98901099 0.96703297]  ****
LogisticRegression()    :
scores :  [0.96703297 0.95604396 0.95604396 1.         0.95604396]
DecisionTreeClassifier()    :
scores :  [0.92307692 0.93406593 0.95604396 0.94505495 0.94505495]
RandomForestClassifier()    :
scores :  [0.98901099 0.97802198 0.95604396 0.96703297 0.95604396]

========================================== StandardScaler()
LinearSVC()    :
scores :  [0.96703297 0.95604396 0.96703297 0.97802198 0.98901099]
SVC()    :
scores :  [0.96703297 0.91208791 0.96703297 0.98901099 1.        ]
LogisticRegression()    :
scores :  [0.97802198 0.97802198 0.93406593 0.98901099 1.        ]  ****
DecisionTreeClassifier()    :
scores :  [0.96703297 0.98901099 0.89010989 0.87912088 0.94505495]
RandomForestClassifier()    :
scores :  [0.97802198 0.96703297 0.92307692 0.97802198 0.97802198]
'''