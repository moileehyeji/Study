from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os
import cv2
import imutils
from keras.preprocessing.image import img_to_array

path = './project/data'
# 학습을 위해 에폭과 초기 학습률, 배치 사이즈, 그리고 이미지의 차원을 초기화합니다
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (64, 64, 3)
OPTI = RMSprop(INIT_LR)

def modeling():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 전결합층
    model.add(Flatten())    # 벡터형태로 reshape
    model.add(Dense(512))   # 출력
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

def callbacks():
    modelpath = path + '/modelcheckpoint/modeling_xy_test_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=10)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=5,factor=0.2 ,verbose=1)
    return er,mo,lr
er,mo,lr = callbacks()

# 분류 대상 카테고리 선택하기 
categories = ["dumbbell","gymball","ladderbarrel","reformer","yogamat","runningmachine","pullupbars"]
nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 64 
image_h = 64

# 데이터 열기 
x_train, x_test, x_val, y_train, y_test, y_val = np.load("./project/data/npy/modeling_xy_val.npy", allow_pickle = True)

# 데이터 정규화하기(0~1사이로)
x_train = x_train.astype("float") / 256
x_test  = x_test.astype("float")  / 256
# print(X_train.shape, y_train.shape) # (3159, 64, 64, 3) (3159, 7)
# print(X_test.shape, y_test.shape)   # (1054, 64, 64, 3) (1054, 7)

# 모델 구조 정의 
model = modeling()
model.compile(loss='categorical_crossentropy', optimizer=OPTI, metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100, batch_size=10, callbacks=[er, lr], validation_data=(x_val, y_val))

# 모델 확인
# print(model.summary())

# 학습 완료된 모델 저장
hdf5_file = path + "/h5/modeling_xy_val.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(x_train, y_train, epochs=100, batch_size=10, callbacks=[er, lr], validation_data=(x_val, y_val))
    model.save_weights(hdf5_file)

# 모델 평가하기 
score = model.evaluate(x_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

# 적용해볼 이미지 
test_image = './project/data/img/test13.jpg'
# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((64,64))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 64, 64,3)
# 예측
pred = model.predict(X)  
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
print('New data category : ',categories[result[0]])


####################################### 사진과 결과 시각화 
print(result)       #[5]
print(result[0])    #5
print(pred.shape)   # (1, 7)
print(pred[0][result[0]])   # 0.9999795

output = cv2.imread(test_image)
output = imutils.resize(output, width=400)
label = "{}: {:.2f}%".format(categories[result[0]], (pred[0][result[0]]) * 100)
cv2.putText(output, label, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
cv2.imshow("Output", output)
cv2.waitKey(0)  # 키 입력 대기 시간 (무한대기)

#######################################


# loss= 2.2184700965881348
# accuracy= 0.5588235259056091
# New data category :  reformer
