# pip install autokeras

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255

# onehot해보자
# onehot안하면 y_train.shape = (60000,) ---> 돌아감
# onehot하면 y_train.shape = (60000,10) ---> 돌아감=====> 결론: 해도되고 안해도 됨
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#=======================================
# autokeras.ImageClassifier(
#     num_classes=None,     
#     multi_label=False,
#     loss=None,
#     metrics=None,                     기본: accuracy
#     project_name="image_classifier",  
#     max_trials=100,                   시도 할 다른 Keras 모델의 최대 수
#     directory=None,                   문자열. 검색 출력을 저장하기위한 디렉토리 경로
#     objective="val_loss",             최소화 또는 최대화 할 모델 메트릭
#     tuner=None,
#     overwrite=False,                  기존 프로젝트가있는 경우 다시로드합니다. 그렇지 않으면 프로젝트를 덮어 씀
#     seed=None,
#     max_model_size=None,
#     **kwargs
# )
#=======================================
model = ak.ImageClassifier(
                            # overwrite=True, 
                            max_trials=1)   # 최대시도:2


#=======================================
# ImageClassifier.fit(
#       x=None, 
#       y=None,         원-핫 인코딩 또는 이진 인코딩이 될 수 있음
#       epochs=None,    최대 1000 Epoch 동안 훈련하지만 ***10 Epoch 동안 유효성 검사 손실이 개선되지 않으면 훈련을 중지
#       callbacks=None, 
#       validation_split=0.2, 
#       validation_data=None, 
#       **kwargs
# )
#=======================================
model.fit(x_train, y_train, epochs=1)

results = model.evaluate(x_test, y_test)

print(results)