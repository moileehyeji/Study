# 모델수정

import numpy as np
import os
import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


path = './project/data'
# 학습을 위해 에폭과 초기 학습률, 배치 사이즈, 그리고 이미지의 차원을 초기화합니다
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (150, 150, 3)
OPTI = Adam(INIT_LR)

def modeling():
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu', input_shape=x_train.shape[1:], padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

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
image_w = 150 
image_h = 150

# 데이터 열기 
x_train = np.load('./project/data/npy/0217_IDG_train_x.npy')
y_train = np.load('./project/data/npy/0217_IDG_train_y.npy')
x_val = np.load('./project/data/npy/0217_IDG_1_val_x.npy')
y_val = np.load('./project/data/npy/0217_IDG_1_val_y.npy')

# 모델 구조 정의 
model = modeling()
model.compile(loss='categorical_crossentropy', optimizer=OPTI, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, callbacks=[er, lr], validation_data=(x_val, y_val))

# # 모델 확인
# # print(model.summary())

# # 학습 완료된 모델 저장
# hdf5_file = path + "/h5/0217_modeling.hdf5"
# if os.path.exists(hdf5_file):
#     # 기존에 학습된 모델 불러들이기
#     model.load_weights(hdf5_file)
# else:
#     # 학습한 모델이 없으면 파일로 저장
#     history =  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, callbacks=[er, lr], validation_data=(x_val, y_val))
#     model.save_weights(hdf5_file)

# 모델 평가하기 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc : ', np.mean(acc))
print('val_acc : ', np.mean(val_acc))

# 적용해볼 이미지 
test_image = './project/data/test13.jpg'
# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((150,150))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 255
X = X.reshape(-1, 150, 150,3)
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

#######################################시각화
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))      

plt.subplot(2,1,1)  #2행 1열중 첫번째
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)  #2행 2열중 두번째
plt.plot(acc, marker='.', c='red')
plt.plot(val_acc, marker='.', c='blue')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['accuracy','val_accuracy'])

plt.show()
#######################################


# acc :  0.6756000839746915
# val_acc :  0.269670328268638

# 결과 안좋음
# val과 의 간격이 가장 좁음