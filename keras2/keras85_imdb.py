# 이진분류 데이터 셋
# 감정 분류 데이터 세트.
# 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1을 부정인 경우 0
# 이미 훈련 데이터와 테스트 데이터를 50:50 비율로 구분해서 제공
# 단어 집합의 크기는 100,000
from tensorflow.keras.datasets import reuters, imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam, Adam, RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#reuters
#num_words: 위에서부터 단어 10000개 가져오겠다.
(x_train, y_train), (x_test, y_test) = imdb.load_data()
                                                    

print(x_train[0], type(x_train[0]))   #<class 'list'>
print(y_train[0], type(y_train[0]))   #1 <class 'numpy.int64'>
print(len(x_train[0]), len(x_train[11]))    #218 99
print('-------------------------------------------------')
print(x_train.shape, x_test.shape)  #(25000,) (25000,)
print(y_train.shape, y_test.shape)  #(25000,) (25000,)

print('뉴스기사의 최대길이 : ', max(len(l) for l in x_train))   #2494
print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train))  #238.71364

''' #시각화
#막대그래프
plt.hist([len(s) for s in x_train], bins=50)
plt.show()  # 1000까지 치우쳐있음 WHY? 가장 많이 나오는 단어가 첫번째임 
'''

#y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y_분포 : ', dict(zip(unique_elements, counts_elements)))
# y_분포 :  {0: 12500, 1: 12500}
print('-------------------------------------------------')


''' #시각화
plt.hist(y_train, bins=46)
plt.show()  #0,1 '''

#x의 단어분포
word_to_index = imdb.get_word_index()
print(word_to_index)
print(len(word_to_index))   #88584  --> word_size즉 input_dim
print(type(word_to_index))  #<class 'dict'>
print('-------------------------------------------------')


#시퀀스화 되어있는 데이터를 문장화
#키와 벨류를 교체
index_to_word = {}  #dintionary
for key, value in word_to_index.items():
    index_to_word[value] = key
#교환 후
print(index_to_word)
print(index_to_word[1])     #the
print(index_to_word[88584]) #'l'
print(len(index_to_word))   #88584


#x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))


#y카테고리 개수 출력
category = np.max(y_train)+1
print('y카테고리 개수 : ', category)    #2

#y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)  #[0 1]

############################전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=1000, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=1000, truncating='pre')
print(x_train.shape, x_test.shape)  #(25000, 500) (25000, 500)


#loss='sparse_categorical_crossentropy'
''' y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)  #(8982, 46) (2246, 46) '''

# 2. 모델구성
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Conv1D, Dropout
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Embedding(input_dim=100000, output_dim=128, input_length=1000))
model.add(LSTM(64, activation='tanh'))
# model.add(Conv1D(64,2, activation='tanh'))
# model.add(Flatten())
# model.add(Dense(32, activation='tanh'))
# model.add(Dense(64, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

re = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1, mode='auto')
er = EarlyStopping(monitor='val_loss', patience=30, mode='auto')

#sparse_categorical_crossentropy(다중 분류 손실함수)
#y값을 원핫인코딩한것돠 동일한 효과를 볼 수 있음
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.2, callbacks=[er])

loss, acc = model.evaluate(x_test, y_test, batch_size=30)
print('loss:', loss)        
print('acc:', acc)  

# loss: 3.304931640625
# acc: 0.8466799855232239


