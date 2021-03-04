# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.
#  NLP 질문
#
# 풍자 데이터 세트에 대한 분류기를 빌드하고 훈련합니다.
# 분류기는 그림과 같이 시그 모이 드에 의해 활성화 된 1 개의 뉴런이있는 최종 레이어를 가져야합니다.
# 네트워크가 이전에 본 적이없는 여러 문장에 대해 테스트됩니다.
# 그리고 그 문장에서 풍자가 올바르게 감지되었는지 여부에 따라 점수가 매겨집니다.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'C:/Study/tf_certificate/Category4/sarcasm.json')
    #json :텍스트 데이터 포맷

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('C:/Study/tf_certificate/Category4/sarcasm.json') as file:
        # 기본적으로 사용하는 함수를  with문 안에 사용하면 되며
        # with문을 나올 때 close를 자동으로 불러줍니다.
        data = json.load(file)
    
    for elem in data:
        sentences.append(elem['headline'])
        labels.append(elem['is_sarcastic'])

    sentences = np.array(sentences) 
    labels = np.array(labels)
    print(sentences.shape, labels.shape)    # (26709,) (26709,)

    x_train = sentences[:training_size]
    y_train = labels[:training_size]
    x_test = sentences[training_size:]
    y_test = labels[training_size:]

    print(len(x_train))

    #문장 토큰화
    token = Tokenizer(num_words=vocab_size, #단어 빈도에 따라 유지할 최대 단어 수
            oov_token=oov_tok)     #word_index에 추가되고 text_to_sequence 호출 중에 어휘 이외의 단어를 대체하는 데 사용
    token.fit_on_texts(sentences)
    print(token.word_index) #603

    #토큰화된 단어 시퀀스 형태
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)

    #패딩
    x_train = pad_sequences(x_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    x_test = pad_sequences(x_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(np.unique(x_train))
    print(len(np.unique(x_train)), len(np.unique(x_test)))    #1000 1000

    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),    #양방향연결
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()


    er = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, mode='auto')
    re = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=10, mode='auto', factor=0.5, verbose=1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[er, re])

    loss = model.evaluate(x_test, y_test)
    print('loss, acc : ', loss)
    # loss, acc :  [0.6610866189002991, 0.8191980719566345]
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category4/mymodel.h5")
