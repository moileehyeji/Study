from tensorflow.keras.preprocessing.text import Tokenizer
# Tokenizer : 텍스트 토큰 화 유틸리티 클래스

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

#어절
token = Tokenizer()
token.fit_on_texts([text])  #문장을 토큰화:texts_to_sequences을 사용하기 전에 필요



# word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환
#{'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}
#빈도수 > 순서순으로 반환
print(token.word_index) 




# 토큰화된 단어들을 시퀀스의 형태로 변환
x = token.texts_to_sequences([text])
print(x)  #[[3, 1, 1, 4, 5, 1, 2, 2, 6]]



# 나는(3)이 진짜(1)의 3배야?
# 아니니까 one hot encoding
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)    #6
x = to_categorical(x)

print(x)
print(x.shape)  #(1, 9, 7)