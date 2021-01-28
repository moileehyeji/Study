# m04_iris 복사
# 실습 : 결과 비교
# 분류모델

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 모델 import
from sklearn.svm import  LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)


# 훈련 loop
scalers = np.array([MinMaxScaler(), StandardScaler()])
models = np.array([LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()])

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
        print('accuracy_score  :', accuracy_score(y_test, y_pred))

        
        print('\n')   



'''
Tensorflow                 :
Dense(Dropout)모델 acc :  0.9912280440330505

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
'''