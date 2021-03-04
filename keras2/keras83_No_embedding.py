from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고네요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '규현이가 잘 생기긴 했어요']


#긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)    #1부터 부여
print(token.word_index) #27개



#단어들을 시퀀스의 형태로 변환
x = token.texts_to_sequences(docs)
print(x)



# 시퀀스의 형태 길이가 일정하지 않음
# pad_sequences : 시퀀스를 동일한 길이로 채움
# max길이에 맞춰서 zero padding
# pad_sequences하기 전에는 1부터 부여되기때문에 1~27(27개)
# pad_sequences한 후 0이 채워지기때문에 0~27(28개) 
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, 
                    padding='pre',      #각 시퀀스 앞(pre)이나 뒤(post)에 0으로 채움
                    maxlen = 5,         #모든 시퀀스의 최대 길이(5)
                    truncating = 'pre'  #시작 또는 끝에서 보다 큰 시퀀스의 값을 제거
                    )
print(pad_x)        #[2, 5] -> [ 0  0  0  2  4]
print(pad_x.shape)  #(13, 5)
#LSTM(13,5,1)
#Dense(13,5)

#np.unique : 1 차원 텐서에서 고유 한 요소를 찾습니다
print(np.unique(pad_x))
print(len(np.unique(pad_x)))    #27 / 0~27(28개)인데 11이 maxlen = 4으로 인해 잘림



# 2. 모델구성
#one hot encoding으로 전처리 할경우 데이터가 너무 커진다.(13,5,28)
#단어 벡터의 크기가 너무 크고 값이 1이 되는 값은 거의 없어 Sparse(희소)한 표현법 -> 압축, 벡터화(input_length)가 필요
#Embedding : 양의 정수 (인덱스)를 고정 된 크기의 조밀한 실수 형태 벡터로 변환
#           거리에 대한 수치로 표현?
#           이 레이어는 모델의 첫 번째 레이어로만 사용할 수 있습니다.
#           input_dim, output_dim 필수
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

pad_x = pad_x.reshape(13,5,1)
#(실습) 임베딩레이어 빼고 모델구성
model = Sequential()
# model.add(LSTM(32, input_shape=pad_x.shape[1:], activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Conv1D(32,2,input_shape=pad_x.shape[1:], activation='relu'))
model.add(Flatten())                        #안해줘도 먹히긴하나, 안해주면 성능이 떨어짐(acc : 0.84 -> 1.0)
# model.add(Dense(32, input_shape=pad_x.shape[1:], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   #3차원 Output Shape 받아들임

#Dense: acc :  0.9230769276618958
#LSTM:  acc :  0.9230769276618958
#Conv1D: acc :  0.9230769276618958


model.summary()
# Embedding layer param# = 308 이해하기
# Output Shape 3차원
'''
model.add(Embedding(input_dim=28,output_dim=11,input_length=5)) # (None, 5, 11) / parameter:308
model.add(Embedding(28,11))                                     #(None, None, 11) / parameter:308
이 둘의 파라미터는 308로 같다. 이는 총 단어의 수 * 내가 지정한 아웃풋 딤 = 28 * 11 = 308이다.
단어의 총 개수가 내가 지정 아웃풋딤의 길이만큼으로 벡터화 된다
'''


# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=50)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

# acc :  1.0

