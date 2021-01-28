# m10_kfold_1 복사
# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test(1/n) 


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
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# n등분 설정
kfold = KFold(n_splits=5, shuffle=True)


#2. 모델구성
model = LinearSVC()

scores = cross_val_score(model, x, y, cv=kfold)     # train : test(x, 1/5)  ,  cv = 5 가능(shuffle이 없기 때문에 따로 정의), 분류:acc, 회귀:r2
print('scores : ', scores)                          # scores :  [0.96666667 0.96666667 1.         0.9        0.96666667]

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


'''