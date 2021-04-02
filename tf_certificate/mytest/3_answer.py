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
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.
# CNN을 사용한 컴퓨터 비전
#
# rock_paper_scissors를 기반으로 Rock-Paper-Scissors에 대한 분류기를 만듭니다.
# TensorFlow 데이터 세트.
#
# 중요 : 최종 레이어는 그림과 같아야합니다. 변경하지 마십시오
# 제공된 코드 또는 테스트가 실패 할 수 있음
#
# 중요 : 이미지는 3 바이트의 색 농도로 150x150으로 테스트됩니다.
# 따라서 입력 레이어가 적절하게 설계되었는지 확인하거나
# 실패 할 수 있습니다.
#
# 이것은 레이블이없는 데이터입니다.
# ImageDataGenerator를 사용하여 자동으로 레이블을 지정할 수 있습니다.
# 그리고 우리는 몇 가지 시작 코드를 제공했습니다.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
        width_shift_range = 0.1,
        height_shift_range= 0.1,
        rescale = 1/255.,
        validation_split= 0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical',
        batch_size = 32,
        subset = 'training'
    )

    test_generator = train_datagen.flow_from_directory(
        directory = TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical',
        batch_size = 32,
        subset = 'validation'
    )


    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Conv2D(128, (5,5), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(5,5),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    es = EarlyStopping(patience = 6)
    lr = ReduceLROnPlateau(factor = 0.25, verbose = 1, patience = 3)

    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit_generator(train_generator, epochs = 1000, validation_data= test_generator,\
         steps_per_epoch= np.ceil(2016/32), validation_steps= np.ceil(504/32), callbacks = [es, lr])

    print(model.evaluate(test_generator, steps = np.ceil(504/32)))
    # [0.080736443400383, 0.9702380895614624]

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
