from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.arange(1, 11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
# print('\n',x, '\n', y)

#2. 모델구성
# ==============================================머신러닝에는 히든레이어가 없다.
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
# model.add(Dense(10))
# model.add(Dense(1))

#3. 컴파일, 훈련
opti = RMSprop(learning_rate=0.01)

model.compile(loss = 'mse', optimizer=opti)
model.fit(x, y, epochs=1000)

# =============================================scatter : 흩어지게하다
y_pred = model.predict(x)

plt.scatter(x, y)                     # 데이터를 파란점으로 표시
plt.plot(x, y_pred, color = 'red')    # 훈련결과 나온 가중치
plt.show()

#4. 평가, 예측
