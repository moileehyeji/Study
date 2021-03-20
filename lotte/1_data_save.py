from PIL import Image
import os, glob
import numpy as np

# ======================================이미지 데이터 읽어 들이기
dir_train = 'C:/Study/lotte/data/train' 
dir_test = 'C:/Study/lotte/data/test' 
X = []
Y = []
TEST = []
nb_classes=1000
image_wh = 255

#train data
for idx in range(1000):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = f'{dir_train}/{idx}'
    files = glob.glob(image_dir +"/*.jpg")
    for i, f in enumerate(files):      
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_wh, image_wh))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
X = np.array(X)
Y = np.array(Y)

#test data
files = glob.glob(dir_test +"/*.jpg")
for i, f in enumerate(files):      
    img = Image.open(f) 
    img = img.convert("RGB")
    img = img.resize((image_wh, image_wh))
    data = np.asarray(img)      # numpy 배열로 변환
    TEST.append(data)

TEST = np.array(TEST)

np.save("C:/Study/lotte/data/npy/1_255_x.npy", arr = X)
np.save("C:/Study/lotte/data/npy/1_255_y.npy", arr = Y)
np.save("C:/Study/lotte/data/npy/1_255_test.npy", arr = TEST)

print(X.shape)  #(48000, 128, 128, 3)
print(Y.shape)  #(48000, 1000)
print(TEST.shape)   #(72000, 128, 128, 3)