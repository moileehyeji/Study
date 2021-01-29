# m04_iris 복사
# KFold             :데이터 n등분 n번 훈련
# cross_val_score   :교차 검증 값
# train : test : validation(1/n)

# 실습 : train, test 나눈 다음에 train만 validation 하지말고, 
# kfold한 후에 train_test_split 사용

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

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)


# 전처리

# n등분 설정
kfold = KFold(n_splits=5, shuffle=False) 

# ================================================================================================KFold.split
# split(X [, y, 그룹])  : 데이터를 학습 및 테스트 세트로 분할하는 인덱스를 생성

for train_index, test_index in kfold.split(x):
    print('================================================================================')
    print("TRAIN:", train_index, "\nTEST:", test_index) 

    # train : test
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]
      
    # train : test : validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77, shuffle = False) 

    print('x_train.shape : ', x_train.shape)        # (96, 4)
    print('x_test.shape  : ', x_test.shape)         # (30, 4)
    print('x_val.shape   : ', x_val.shape)          # (24, 4)

    print('y_train.shape : ', y_train.shape)        # (96, )
    print('y_test.shape  : ', y_test.shape)         # (30, )
    print('y_val.shape   : ', y_val.shape)          # (24, )

    # print('x_train.shape : \n', x_train)              
    # print('x_test.shape  : \n', x_test)               
    # print('x_val.shape   : \n', x_val)   

    # print('y_train.shape : \n', y_train)              
    # print('y_test.shape  : \n', y_test)
    # print('y_val.shape   : \n', y_val)

    
    # 훈련마다 평가
    #2. 모델구성
    model = SVC()

    scores = cross_val_score(model, x_train, y_train, cv=kfold)     
    print('scores : ', scores)    #가장 좋은 결과 :  scores :  [1. 1. 1. 1. 1.]                     
    


#2. 모델구성
    # model = SVC()

    scores = cross_val_score(model, x_train, y_train, cv=kfold)     # train : test : validation(x_train 1/5)
    print('scores : ', scores)                                      # scores :  [1. 1. 1. 1. 1.]

           




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
