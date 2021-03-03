# 전이학습관련 파라미터 참고: https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet',       #ImageNet에 대해 사전 훈련된 가중치
            include_top=False,          #False일때 input_shape 변경가능 
            input_shape = (32,32,3))    
# print(model.weights)  # layer 13개


model.trainable = True  #(기본값True)
model.summary()
print(len(model.weights))           #26(weight와 bias가 존재하는 layer:13개)
print(len(model.trainable_weights)) #26(weight와 bias가 존재하는 layer:13개)
""" 
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0 
"""

model.trainable = False # 제가 훈련 안 시킬고에요 가중치만 주세요(기본값True)
                        # 가져온 가중치에서 수정이 시작
                        # False:훈련 중 지정된 층의 가중치가 업데이트되지 않음
model.summary()
print(len(model.weights))           #26  
print(len(model.trainable_weights)) #0 
""" 
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688 
"""




