import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# 1. Adam
# optimizer = Adam(lr=0.1)
# loss :  1.7184653700041963e-10 결과물 :  [[10.999966]]
# optimizer = Adam(lr=0.01)
# loss :  0.04318828135728836 결과물 :  [[10.653315]]
# optimizer = Adam(lr=0.001)
# loss :  7.409291236104565e-11 결과물 :  [[10.999991]]
# optimizer = Adam(lr=0.0001)
# loss :  2.3693104594713077e-05 결과물 :  [[10.998213]]  -> epochs 부족

# 2. Adadelta
# optimizer = Adadelta(lr = 0.1)
# loss :  0.05831011384725571 결과물 :  [[11.4341135]]
# optimizer = Adadelta(lr = 0.01)
# loss :  0.000529905897565186 결과물 :  [[11.033939]]
# optimizer = Adadelta(lr = 0.001)
# loss :  10.94493293762207 결과물 :  [[5.081273]]
# optimizer = Adadelta(lr = 0.0001)
# loss :  37.080177307128906

# 3. Adamax
# optimizer = Adamax(lr = 0.1)
# loss :  39.97328567504883 결과물 :  [[7.645391]]
optimizer = Adamax(lr = 0.01)
# loss :  8.39897075499696e-12 결과물 :  [[11.000003]]
# optimizer = Adamax(lr = 0.001)
# loss :  9.030532055476215e-07 결과물 :  [[10.998]]
# optimizer = Adamax(lr = 0.0001)
# loss :  0.0038345367647707462 결과물 :  [[10.924022]]

# 4. Adagrad
# optimizer = Adagrad(lr = 0.1)
# loss :  55692.7265625 결과물 :  [[331.66443]]
# optimizer = Adagrad(lr = 0.01)
# loss :  4.448892809705285e-09 결과물 :  [[11.000096]]
# optimizer = Adagrad(lr = 0.001)
# loss :  1.0334815669921227e-05 결과물 :  [[11.001784]]
# optimizer = Adagrad(lr = 0.0001)
# loss :  0.00726682785898447 결과물 :  [[10.892901]]

# 5. RMSprop
# optimizer = RMSprop(lr = 0.1)
# loss :  277820672.0 결과물 :  [[-35848.004]]
# optimizer = RMSprop(lr = 0.01)
# loss :  0.261788547039032 결과물 :  [[9.969362]]
# optimizer = RMSprop(lr = 0.001)
# loss :  0.001047688303515315 결과물 :  [[10.984278]]
# optimizer = RMSprop(lr = 0.0001)
# loss :  0.0015332532348111272 결과물 :  [[11.065333]]

# 5. SGD
# optimizer = SGD(lr = 0.1)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr = 0.01)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr = 0.001)
# loss :  6.836998181825038e-06 결과물 :  [[11.001393]]
# optimizer = SGD(lr = 0.0001)
# loss :  0.001129663665778935 결과물 :  [[10.958827]]

# 5. Nadam
# optimizer = Nadam(lr = 0.1)
# loss :  1248.8984375 결과물 :  [[60.975517]]
# optimizer = Nadam(lr = 0.01)
# loss :  2961.343017578125 결과물 :  [[-102.695335]]
# optimizer = Nadam(lr = 0.001)
# loss :  3.4845015327994444e-12 결과물 :  [[11.000002]]
# optimizer = Nadam(lr = 0.0001)
# loss :  2.07630200748099e-05 결과물 :  [[10.9914875]]


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mae = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print('loss : ', loss, '결과물 : ', y_pred)


