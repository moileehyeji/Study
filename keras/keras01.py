import numpy as np
import tensorflow as tf

#딥러닝

#1. 데이터
#데이터이상수치는 데이터 수치로만 판단하면 안됨 
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
# from: 폴더구조와 같이 내부에 있는 Sequential을 쓰겠다.
# Sequential: 순차적 신경망 모델을 구성
# Dense: 기본적인 DNN또는 ANN모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#순차적인 신경망모델  
#input_dim: 해당 모델의 input layer(위:input,아래:output)
#activation: 활성화함수 (기본값존재:linear)
model = Sequential()
model.add(Dense(5,input_dim=1, activation='linear')) #하이퍼 파라미터 튜닝
model.add(Dense(3,activation='linear'))
model.add(Dense(4))
model.add(Dense(1)) #output

#3. 컴파일, 훈련
#훈련할때마다 값이 다른 것은 가중치저장으로 해결할 수 있다. 
model.compile(loss='mse', optimizer='adam') #adam을 이용해 loss를 최소화하여 최적의 W를 구하기
model.fit(x,y,epochs=100, batch_size=1) #1개씩 잘라서 100번 훈련(기울기값W 구하기)(batch_size가 크면 속도는 빠르지만 성능은 안좋다)

#4. 평가, 예측
#loss값이 낮을수록 좋다
#model.predict(예측하고싶은 값)
loss = model.evaluate(x,y,batch_size=1) #fit에서 가중치W 이미 생성이 되었는데 같은 데이터로 평가->나중에는 훈련데이터,평가데이터 분리할 것
print('loss : ', loss)

result = model.predict([4]) #1~3훈련, 4예측해봐
result2 = model.predict(x) #x에 대해 예측된 y값
x_pred= np.array([4])
result3=model.predict(x_pred)
#fit으로 생성된 가중치W로 값을 예측
print('result : ', result)
print('result2 : ', result2)
print('result3 : ', result3)


