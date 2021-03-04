# 이미지는 
# C:\data\Image\vgg에 4개 넣기
# 강아지, 고양이, 라이언, 슈트
# 파일명
# dog1.jpg, cat1.jpg
# lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 1.데이터
path = 'C:/data/Image/vgg'
img_dog = load_img(f'{path}/dog1.jpg', target_size=(224,224))
img_cat = load_img(f'{path}/cat1.jpg', target_size=(224,224))
img_lion = load_img(f'{path}/lion1.jpg', target_size=(224,224))
img_suit = load_img(f'{path}/suit1.jpg', target_size=(224,224))

# print(img_suit) # <PIL.Image.Image image mode=RGB size=224x224 at 0x265EF2D85B0>

# plt.imshow(img_cat)
# plt.show()

# 이미지 수치화
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(arr_lion)
# print(type(arr_dog))    #<class 'numpy.ndarray'>
# print(arr_dog.shape)    #(224, 224, 3) -> VGG16 기본값?

#RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
# VGG16 전이모델에 맞춰 preprocessing해준다
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
# print(type(arr_dog))    #<class 'numpy.ndarray'>
# print(arr_dog.shape)    #(224, 224, 3) -> VGG16 기본값?

# 이미지이므로 4차원으로 훈련시켜야 한다
# 4개 이미지 합하면? (4, 224, 224, 3) 4차원
# np.stack: 새 축을 따라 일련의 배열을 결합(합치려는 배열들의 shape이 전부 동일해야함)
arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
# print(arr_input.shape)  #(4, 224, 224, 3)

# 2.모델구성
# 우리 훈련안시키고 결과만 볼거야
model = VGG16()
results = model.predict(arr_input)

print(results)  #수치화된 이미지 결과
print('results.shape : ', results.shape)    #results.shape :  (4, 1000), 1000: ImageNet에서 분류할수 있는 category 수

#results(수치화된 이미지 결과) 확인하기
# decode_predictions : ImageNet 모델의 예측을 디코딩
# 반환 : 최고 수준의 예측 튜플 목록. 
#        일괄 입력에서 샘플 당 하나의 튜플 목록. (class_name, class_description, score)
# pred배열의 모양이 잘못된 경우 (2D 여야 함).
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print('--------------------------------------------------------')
print('results[0] : ', decode_results[0])#miniature_poodle 73%
print('--------------------------------------------------------')
print('results[1] : ', decode_results[1])#sleeping_bag     16%
print('--------------------------------------------------------')
print('results[2] : ', decode_results[2])#envelope         14%
print('--------------------------------------------------------')
print('results[3] : ', decode_results[3])#bassoon          21%
print('--------------------------------------------------------')
