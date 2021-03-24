# tf.train.checkpoint vs latest_checkpoint

# Checkpoint 은 학습된 모델의 Variable 값을 저장하는 파일이다. 
# Checkpoint 파일을 저장하고 불러옴으로써 학습된 모델을 재사용하고, 
# 지난 학습을 이어서 더 하고 하는 작업들이 가능해진다.

# ===============================tf.train.Checkpoint( root=None, **kwargs )
'''
    - tf.keras.optimizers.Optimizer 구현
    - tf.Variable
    - tf.data.Dataset 반복기
    - tf.keras.Layer 구현 또는 tf.keras.Model
    - 위의 4가지와 같이 추적 가능한 상태를 포함하는 유형인 값을 갖는 키워드 인수를 허용
    - 이 값을 체크 포인트와 함께 저장
    - 체크 포인트 번호 매기기를 위한 save_counter를 유지
    - TensorFlow 1세대의 tf.compat.v1.train.Saver는 variable.name 기반 체크 포인트를 쓰고 읽음
    - Checkpoint.save() 및 Checkpoint.restore()는 객체 기반 체크 포인트를 쓰고 읽음
    <code 예시>
    checkpoint = tf.train.Checkpoint(model)
    save_path = checkpoint.save('/tmp/training_checkpoints')
    # ~ ㄴcheckpoint
    # ~ training_checkpoints.ckpt.data-00000-of-00001
    # ~ training_checkpoints.ckpt.index
    # ~ training_checkpoints.ckpt.meta
    checkpoint.restore(save_path)
'''

# ==========tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
'''
    - 제공된 checkpoint_dir에서 체크포인트 상태를 가져오고 해당 TensorFlow2(선호) 또는 TensorFlow1.x 체크 포인트 경로를 찾음
    - latest_filename 인수는 v1.Saver.save를 사용하여 체크포인트를 저장하는 경우에만 적용
    - 최신 검사점의 전체경로 또는 검사점이없는 경우 None 반환
    <code 예시>
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
'''


import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 모델 정의
# 간단한 Sequential 모델을 정의합니다
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# 모델 객체를 만듭니다
model = create_model()

# 모델 구조를 출력합니다
model.summary()