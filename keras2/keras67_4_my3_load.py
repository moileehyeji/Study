# 나를 찍어서 내가 남자인지 여자인지에 대해 결과

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model("../data/h5/[0.52]k67_4_model.h5")


# img = cv2.imread("../data/Image/gender/suji.png")
img = cv2.imread("../data/Image/gender/hyeji.jpg")
# img = cv2.imread("../data/Image/gender/youngri.jpg")
img = cv2.resize(img,dsize=(32,32))/255.0

result = model.predict(np.array([img]))
acc = (1-result[0][0])*100
print('%.2f' %acc, '%')
result = np.where(result>0.5, 'male', 'female')
print(result)


plt.imshow(img)
plt.show()

""" 
# seoyoung's
# model = load_model("../data/h5/[0.58]k67_4_model.h5")
model = load_model("../data/h5/[0.52]k67_4_model.h5")

img = cv2.imread("../data/Image/gender/hyeji.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img,dsize=(32,32))/255.0

result = model.predict(np.array([img]))

# print(result) # [[0.3412144]] 괄호 없애주기---> [0][0]
if result>0.5:
    print('\n Accuracy: %.4f' % (result[0][0]),'의 정확도를 가진다')
    print("따라서 그는 남자다")
else:
    print('\n Accuracy: %.4f' % (1-result[0][0]),'의 정확도를 가진다')
    print( "따라서 그녀는 여자다")
plt.imshow(img)
plt.show() """