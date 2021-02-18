from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split


# ======================================분류 대상 카테고리 선택하기 
dir = "./project/data/img"
categories = ["dumbbell","gymball","ladderbarrel","reformer","yogamat","runningmachine","pullupbars"]
nb_classes = len(categories)

# =====================================이미지 크기 지정 
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3

# ======================================이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, value in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = dir + "/" + value
    files = glob.glob(image_dir +"/*.jpg")
    for i, f in enumerate(files):      
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
# 이미지를 RGB로 변환 후, 64x64 크기로 resize
X = np.array(X)
Y = np.array(Y)
# print(X.shape, Y.shape)     #   (4213, 64, 64, 3) (4213, 7)


# ======================================학습 전용 데이터와 테스트 전용 데이터 구분 test
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle = True, random_state = 66)
xy = (X_train, X_test, y_train, y_test) # tuple
# print(X_train.shape, y_train.shape) # (3159, 64, 64, 3) (3159, 7)
# print(X_test.shape, y_test.shape)   # (1054, 64, 64, 3) (1054, 7)

print('>>> data 저장중 ...')
np.save("./project/data/npy/modeling_xy_test.npy", xy)
print("ok,", len(Y))

# ======================================학습 전용 데이터와 테스트 전용 데이터 구분 validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)
xy = (X_train, X_test, X_val, y_train, y_test, y_val) # tuple
print(X_train.shape, y_train.shape) # (2696, 64, 64, 3) (2696, 7)
print(X_test.shape, y_test.shape)   # (843, 64, 64, 3) (843, 7)
print(X_val.shape, y_val.shape)   # (674, 64, 64, 3) (674, 7)

print('>>> data 저장중 ...')
np.save("./project/data/npy/modeling_xy_val.npy", xy)
print("ok,", len(Y))