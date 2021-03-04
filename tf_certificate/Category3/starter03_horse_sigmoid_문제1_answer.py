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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly
# CNN을 사용한 컴퓨터 비전
#
# 제공된 데이터를 사용하여 말 또는 인간에 대한 분류기를 만들고 훈련시킵니다.
# 최종 레이어가 그림과 같이 시그 모이 드에 의해 활성화 된 1 개의 뉴런인지 확인합니다.
#
#이 테스트는 300x300 크기의 3 바이트 색상 심도를 사용하므로
# 그에 따라 신경망 설계

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'C:/Study/tf_certificate/Category3/horse-or-human.zip')
    local_zip = 'C:/Study/tf_certificate/Category3/horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('C:/Study/tf_certificate/Category3/tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'C:/Study/tf_certificate/Category3/testdata.zip')
    local_zip = 'C:/Study/tf_certificate/Category3/testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('C:/Study/tf_certificate/Category3/tmp/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)
        rescale = 1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        #Your Code Here
        'C:/Study/tf_certificate/Category3/tmp/horse-or-human',
        target_size=(300,300),
        batch_size = 1500,
        class_mode = 'binary')

    validation_generator = validation_datagen.flow_from_directory(
        #Your Code Here
        'C:/Study/tf_certificate/Category3/tmp/testdata',
        target_size=(300,300),
        batch_size = 1500,
        class_mode = 'binary')

    x_train = train_generator[0][0]
    y_train = train_generator[0][1]
    x_val = validation_generator[0][0]
    y_val = validation_generator[0][1]


    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        tf.keras.layers.Conv2D(64, (5,5), activation='relu',input_shape=(300, 300, 3)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
        # tf.keras.layers.Dense(2, activation='softmax')

    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    er = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='auto')
    re = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.5, verbose=1)
    model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[er, re], validation_split=0.2)

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.
    # 참고 : 교육에 시간이 너무 오래 걸리는 경우 배치 크기 설정을 고려해야합니다.
    # 생성기에 적절하게 적용하고 model.fit () 함수의 epoch 당 단계.

    loss = model.evaluate(x_val, y_val)
    print('loss, acc : ', loss)
    # [sigmoid] loss, acc :  [2.4095427989959717, 0.6640625]
    # [softmax] loss, acc :  [2.5317208766937256, 0.76953125]

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category3/mymodel_2.h5")