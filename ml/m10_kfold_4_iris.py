# m04_iris 복사
# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test : validation(1/n)

# 모델별 비교

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
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
dataset = load_iris()
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
Tensorflow                 :    이게 이겨야 돼
Dense, LSTM, Conv1D 모델 acc :  1.0

======================================================================================1. 기본

                    <MinMaxScaler>
1. LinearSVC               :  
    예측값 :  [1 2 1 1 2 0 2 2 2 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  0.9333333333333333
    accuracy_score  :  0.9333333333333333 

2. SVC                     : 
    예측값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  1.0
    accuracy_score  :  1.0        

3. KNeighborsClassifier    :
    예측값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  1.0
    accuracy_score  :  1.0

4. LogisticRegression      :
    예측값 :  [1 2 2 1 2 0 2 2 2 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy :  0.9
    accuracy_score :  0.9

5. DecisionTreeClassifier  :
    예측값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  1.0
    accuracy_score  :  1.0

6. RandomForestClassifier  :
    예측값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  1.0
    accuracy_score  :  1.0



                     <StandardScaler>

1. LinearSVC               :  
    예측값 :  [1 2 1 1 2 0 2 1 2 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy        :  0.9666666666666667
    accuracy_score  :  0.9666666666666667

2. SVC                     : 동일    

3. KNeighborsClassifier    : 동일

4. LogisticRegression      :
    예측값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    실제값 :  [1 2 1 1 2 0 2 1 1 2 0 1 0 1 1 0 0 0 0 0 2 0 2 0 1 2 1 0 0 0]
    accuracy :  1.0
    accuracy_score :  1.0

5. DecisionTreeClassifier  : 동일

6. RandomForestClassifier  : 동일


======================================================================================2. 기본 + Kfold

========================================== MinMaxScaler()
LinearSVC()    :
scores :  [0.83333333 1.         0.91666667 0.875      0.95833333]
SVC()    :
scores :  [1.         1.         0.95833333 0.95833333 1.        ]  ****
LogisticRegression()    :
scores :  [0.91666667 0.91666667 0.875      0.875      0.79166667]
DecisionTreeClassifier()    :
scores :  [1.         0.95833333 0.95833333 1.         0.95833333]
RandomForestClassifier()    :
scores :  [1.         1.         0.91666667 1.         0.95833333]

========================================== StandardScaler()
LinearSVC()    :
scores :  [1.         0.95833333 0.91666667 0.95833333 0.95833333]
SVC()    :
scores :  [1.         0.95833333 1.         1.         0.95833333] ****
LogisticRegression()    :
scores :  [0.95833333 1.         0.95833333 1.         1.        ] ****
DecisionTreeClassifier()    :
scores :  [1.         1.         0.95833333 0.95833333 1.        ]
RandomForestClassifier()    :
scores :  [0.95833333 1.         0.95833333 0.95833333 1.        ]
'''