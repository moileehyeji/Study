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
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.
# 질문
#
# 다음을 사용하여 흑점 활동을 예측하는 신경망을 구축하고 훈련합니다.
# Sunspots.csv 데이터 세트.
#
# 신경망은 정규화 된 데이터 세트에서 MAE가 0.12 이하 여야합니다.
# 상위 마크입니다.
#
# 데이터 정규화를위한 코드가 제공되며 변경해서는 안됩니다.
#
#이 파일의 맨 아래에 몇 가지 테스트가 있습니다.
# 모델을 확인하려는 경우 코드.

# 참고 : 모델에서 람다 레이어를 사용하지 마십시오. 지원되지 않습니다.
# 채점 인프라.


import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # tf.data.Dataset은 대량의 데이터를 표현할 수 있는 API
    series = tf.expand_dims(series, axis=-1)
    # expand_dims : 배열의 axis번째 차원을 늘려줌
    # expand_dims :  tf.Tensor(
    # [[0.24284279]
    # [0.26192868]
    # [0.29306881]
    # ...
    # [0.19713712]
    # [0.24434957]
    # [0.29934706]], shape=(3000, 1), dtype=float64)
    # print('expand_dims : ', series)
    ds = tf.data.Dataset.from_tensor_slices(series)
    # from_tensor_slices: np.array -> tf.Dataset
    # <types: tf.float64>
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # window : shift씩 이동하여 (window_size(x)+1(y))만큼 그룹화 (drop_remainder=True 남은부분 버리자)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # flat_map : lambda함수를 맵핑해 (window_size+1)만큼 읽어들인 뒤 단일 dataset으로 반환
    # window함수에서 31개씩 그룹화된 데이터를 1차원으로 쭉 나열
    ds = ds.shuffle(shuffle_buffer)
    #shuffle : buffer_size(=shuffle_buffer개의) 인덱스 사이에서 랜덤 추출하여 shuffle
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    #map:함수를 맵핑, train(w[:-1])/label(w[1:]) 이렇게 x, y로 공급해준다
    #w[:-1], w[1:]: 마지막 하루 뺀 것 train data로, 첫날 하루 뺀 것 label로 활용
    return ds.batch(batch_size).prefetch(1)
    #batch_size(32)만큼 1세트씩 미리 데이터를 fetch해준다
    #전체데이터를 미리 만들어 학습시키는 것이 아니라 병렬처리하기 때문에 학습속도 향상


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'C:/Study/tf_certificate/Category5/sunspots.csv')

    time_step = []
    sunspots = []

    with open('C:/Study/tf_certificate/Category5/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    # print(sunspots)
    # print(time_step)
    # print(type(sunspots), type(time_step)) #<class 'list'>

    series = np.array(sunspots)
    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    print(series)
    print(series.shape) #(3235,)
    # [0.24284279 0.26192868 0.29306881 ... 0.03314917 0.03992968 0.00401808]

    # The data should be split into training and validation sets at time step 3000
    # 데이터는 3000 단계에서 학습 및 검증 세트로 분할되어야합니다.
    # DO NOT CHANGE THIS CODE
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid =time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000


    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    valid_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    print('==========',np.array(train_set).shape)
    model = tf.keras.models.Sequential([
        # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
        # 여기에 귀하의 코드. 첫 번째 레이어가 무엇이든 선택한 레이어 유형에 따라 위의 Windowed_dataset를 사용할 때 입력 모양은 [None, 1]이됩니다.
        tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    # 훈련하는 동안이 텍스트를 본다면주의하십시오-무시해도 안전합니다
    # BaseCollectiveExecutor :: StartAbort 범위를 벗어남 : 시퀀스 끝
    # [[{{node IteratorGetNext}}]]


    # # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    model.compile(loss='mae', optimizer='adam')
    model.fit(train_set, batch_size=256, epochs=100)

    loss = model.evaluate(valid_set)
    print('mae : ', loss)
    # mae :  0.038475774228572845



    # THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
    # BEFORE UPLOADING YOU CAN DO IT WITH THIS
    # 이 코드는 예측을 위해 테스터에서 사용됩니다. 모델을 테스트하려면
    # 업로드하기 전에 이것으로 할 수 있습니다
    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    import math

    window_size = 30
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

    # To get the maximum score, your model must have an MAE OF .12 or less.
    # When you Submit and Test your model, the grading infrastructure
    # converts the MAE of your model to a score from 0 to 5 as follows:
    # 최대 점수를 얻으려면 모델의 MAE가 .12 이하 여야합니다.
    # 모델을 제출하고 테스트 할 때 채점 인프라
    # 모델의 MAE를 다음과 같이 0에서 5까지의 점수로 변환합니다.

    test_val = 100 * result
    score = math.ceil(17 - test_val)
    if score > 5:
        score = 5

    print(score)
    # 5
    return model 


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category5/mymodel.h5")



