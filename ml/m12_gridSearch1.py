# m10_kfold_1 복사
# gridSearch : 특정모델 파라미터 자동화 (격자형으로 촘촘하게 파라미터를 다 쓰겠다)
# 느림


import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 모델 import
from sklearn.svm import  LinearSVC, SVC

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

# n등분 설정
kfold = KFold(n_splits=5, shuffle=True)


#   =========================================================================================================GridSearchCV
# parameters        : dictionary 형태      -->  GridSearchCV의 모델의 파라미터 튜닝
# GridSearchCV      : 이 자체가 모델
# best_estimator_   : 자동으로 최적의 가중치로 훈련한 예측값 출력

# parameters       
parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},                              # 4번 훈련
    {'C':[1, 10, 100],       'kernel':['rbf'],     'gamma':[0.001, 0.0001]},    # 6번 훈련
    {'C':[1, 10, 100, 100],  'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}     # 8번 훈련      : total 18번
]

#2. 모델구성
# model = SVC()

# GridSearchCV : 이 자체가 모델
model = GridSearchCV(SVC(), parameters, cv=kfold)       #(모델, 파라미터, kfold)    : 18(파라미터) * 5(kfold) = 90번 훈련


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# best_estimator_
print('최적의 매개변수 : ', model.best_estimator_)        # 최적의 매개변수 :  SVC(C=1000, kernel='linear')

y_pred = model.predict(x_test)                           # 자동으로 최적의 가중치로 훈련한 예측값 출력
print('최종 정답률 : ', accuracy_score(y_test, y_pred))   # 최종 정답률 :  1.0

aaa = model.score(x_test, y_test)
print(aaa)                                               # 1.0

#   =========================================================================================================

# scores = cross_val_score(model, x, y, cv=kfold)    
# print('scores : ', scores)                         



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