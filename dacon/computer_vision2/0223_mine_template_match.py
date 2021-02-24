# 이미지 자르기 시도1

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 노이즈부터 제거하자 (한수오빠 dacon_img_change.py 참고)
# rect = (1,1,img.shape[0]-1,img.shape[1]-1)
file_path = 'C:/Study/dacon/computer2/data/dirty_mnist_2nd/00003.png'

image = cv2.imread(file_path) # cv2.IMREAD_GRAYSCALE
image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
image2 = np.where((image <= 254) & (image != 0), 0, image)
image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
image_data = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
# image_data = cv2.resize(image_data, (128, 128))
image_data = np.array(image_data)
image_data = image_data.astype(np.uint8)

# plt.imshow(image_data),plt.colorbar(),plt.show()


# 템플릿매칭으로 찾아보자
# 탬플릿 매칭은 그레이스케일 이미지를 사용
file_path = "C:/Study/dacon/computer/data/img/images_train/6/108.png"

templit = cv2.imread(file_path) # cv2.IMREAD_GRAYSCALE
templit = cv2.cvtColor(templit, cv2.IMREAD_GRAYSCALE)
templit2 = np.where((templit <= 100) & (templit != 0), 0, templit/templit*255)
templit3 = cv2.erode(templit2,kernel=np.ones((2, 2), np.uint8), iterations=1)
templit_data = np.array(templit3)
templit_data = templit_data.astype(np.uint8)
# templit_data = cv2.resize(templit_data, (32,32))
plt.imshow(templit_data),plt.colorbar(),plt.show()

# print(image_data.shape, templit_data.shape)#(256, 256, 4) (28, 28, 4)

src = image_data
templit = templit_data
dst = image_data

result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED) # 템플릿 매칭을 적용
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
x, y = minLoc
h, w = templit.shape[:2]
dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1)

plt.imshow(dst),plt.colorbar(),plt.show()