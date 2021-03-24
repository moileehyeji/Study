import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import cv2


# 오픈 cv를 통해 전처리 후 128, 128로 리사이징 npy 저장!

img=[]
img_y=[]
for i in range(1000):
    for de in range(48):
        filepath='C:/Study/lotte/data/train/'+str(i)+'/'+str(de)+'.jpg'
        #image=Image.open(filepath)
        #image_data = image.resize((128,128))
        image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
        # 커널 생성(대상이 있는 픽셀을 강조)
        image = cv2.resize(image, (224, 224))

        image_data = np.array(image)
        print(i)
        img.append(image_data)
        img_y.append(i)

np.save('C:/Study/lotte/data/npy/224_project_x.npy', arr=img)
np.save('C:/Study/lotte/data/npy/224_project_y.npy', arr=img_y)
x = np.load("C:/Study/lotte/data/npy/224_project_x.npy",allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/224_project_y.npy",allow_pickle=True)
print(x.shape, y.shape)

print("train 끝")
''' 
img=[]
for i in range(36000):
    filepath='C:/Study/lotte/data//test/'+str(i)+'.jpg'
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
    image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE

    image = cv2.resize(image, (224, 224))
    image_data = np.array(image)
    print(i)
    img.append(image_data)

np.save('C:/Study/lotte/data/npy/224_test1.npy', arr=img)


print("predict 끝")
test1 = np.load("C:/Study/lotte/data/npy/224_test1.npy",allow_pickle=True)
print(test1.shape)



img=[]
for i in range(36000,72000):
    filepath='C:/Study/lotte/data/test/'+str(i)+'.jpg'
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
    image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
    
    image = cv2.resize(image, (224, 224))
    image_data = np.array(image)
    print(i)
    img.append(image_data)

np.save('C:/Study/lotte/data/npy/224_test2.npy', arr=img)


print("predict 끝") 
test2 = np.load("C:/Study/lotte/data/npy/224_test2.npy",allow_pickle=True)
print(test2.shape)  #(36000, 224, 224, 3)
 '''