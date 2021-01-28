'''
첫번째 암흑기(1974-1980)
퍼셉트론은 AND 또는 OR 같은 선형 분리가 가능한 문제는 가능하지만,  
선형(linear) 방식으로 데이터를 구분할 수 없는 XOR문제에는 적용할 수 없다는 것을 수학적으로 증명했다. 
이에 따라 인공지능에 대한 대규모 연구는 중단되어 다시 한번 암흑기에 접어들게 된다.
'''

# 딥러닝으로 구현

import numpy as np
from sklearn.svm import LinearSVC   # 선형으로 갈라주는 모델
from sklearn.svm import SVC         # LinearSVC의 개선모델
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델구성
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_data, y_data, epochs=100, batch_size=1)   

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, '의 예측결과 : ', y_pred)     # [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [[0.5241226] [0.2723666] [0.5109486] [0.2620362]]

result = model.evaluate(x_data, y_data)    
print('model.score : ', result[1])         # model.score :  0.5

# acc = accuracy_score(y_data, y_pred)
# print('accuracy_score : ', acc)         #ValueError: Classification metrics can't handle a mix of binary and continuous targets    