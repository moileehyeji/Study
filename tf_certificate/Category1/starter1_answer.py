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
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
# 이 시험에는 1-5 개에서 난이도가 증가하는 5 개의 문제가 있습니다.
# 질문에 대한 등급의 가중치는 상대적입니다.
# 난이도. 따라서 카테고리 1 문제는 상당한 점수를 얻습니다.
# 카테고리 5 질문보다 적습니다.
#
# 모델에서 람다 레이어를 사용하지 마십시오.
# 질문을 해결하는 데 필요한 것은 아닙니다.
# Lambda 계층은 그레이딩 인프라에서 지원되지 않습니다.
#
# 모델을 제출하려면 모델 제출 및 테스트 버튼을 사용해야합니다.
# 최종적으로 시험을 제출하기 전에이 카테고리에서 한 번 이상,
# 그렇지 않으면이 카테고리에서 0 점을 받게됩니다.
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# 시작하기 질문
#
# 이 데이터가 주어지면 xs를 ys와 일치하도록 신경망을 훈련시킵니다.
# X의 새로운 값에 대한 예측자는 float 값을 제공합니다.
# 원하는 답변에 매우 가깝습니다.
# 즉, print (model.predict ([10.0]))는 만족스러운 결과를 제공합니다.
# 테스트 인프라는 다음을 수용하는 훈련 된 모델을 기대합니다.
# [1]의 입력 형태

import numpy as np

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # YOUR CODE HERE
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(32, input_shape =(1,), activation='linear'))
    model.add(Dense(64, activation='linear'))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(xs, ys, epochs=100, batch_size=20, validation_split=0.2)


    loss = model.evaluate(xs, ys)
    pred = model.predict([10.0])

    print('loss : ', loss)
    print('xs[10.0] : ', pred)
    # loss :  [9.677754860604182e-05, 0.008350747637450695]
    # xs[10.0] :  [[10.9585905]]
    
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category1/mymodel.h5")
