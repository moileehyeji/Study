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
    urllib.request.urlretrieve(url, 'C:/Study/tf_certificate/Category3/rps.zip')# URL로 표시된 네트워크 객체를 로컬 파일로 복사
    local_zip = 'C:/Study/tf_certificate/Category3/rps.zip' 
    zip_ref = zipfile.ZipFile(local_zip, 'r')#ZIP 파일을 오픈, 'r':기존파일 읽기
    zip_ref.extractall('C:/Study/tf_certificate/Category3/tmp/')#압축해제
    zip_ref.close()#아카이브 파일을 닫기

    
    TRAINING_DIR = "C:/Study/tf_certificate/Category3/tmp/rps/"
    training_datagen = ImageDataGenerator(
            # YOUR CODE HERE
            rescale= 1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.4)


    # YOUR CODE HERE
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                        target_size = (150,150),        
                                                        batch_size = 3000,
                                                        class_mode = 'categorical',     
                                                        subset='training')

    val_datagen = ImageDataGenerator(rescale= 1./255, validation_split=0.4)
    val_generator = val_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size=(150,150),
                                                    batch_size=3000,
                                                    class_mode='categorical',
                                                    subset = 'validation')


    x_train = train_generator[0][0]
    y_train = train_generator[0][1]
    x_val = val_generator[0][0]
    y_val = val_generator[0][1]

    from sklearn.model_selection import train_test_split
    x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size = 0.5, shuffle = True, random_state = 66)

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(64, (5,5), activation='relu',input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dense(3, activation='softmax')
    ])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    er = EarlyStopping(monitor='val_loss', patience=15, mode='auto')
    re = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.5, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=200, batch_size=32, callbacks=[er, re], validation_data=(x_val, y_val))

    loss = model.evaluate(x_test, y_test) 
    print('loss, acc : ', loss)
    # loss, acc :  [2.304771661758423, 0.7658730149269104]

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category3/mymodel.h5")
