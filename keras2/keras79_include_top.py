# transfer learning과 유사하게 보일 수도 있지만 
# pre training된 weight을 내려받어 통과시킨 feature을 그대로 사용하면서 
# 학습시 fully connected계층만 트레이닝 시키는 transfer learning과는 명백히 다릅니다
# include_top은 전체 VGG16의 마지막 층, 즉 분류를 담당하는 곳을 불러올지 말지를 정하는 옵션
# 기본값 true로 불러옴
# summary 확인할것

from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True, input_shape = (224,224,3)) 
# model = VGG16() 


model.trainable = False
model.summary()
print(len(model.weights))           
print(len(model.trainable_weights)) 

# VGG16(include_top=False)
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________
# 26
# 0

# VGG16()   = include_top=True과 결과가 동일
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________
# 32
# 0

