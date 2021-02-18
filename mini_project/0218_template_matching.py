# cv2.imread(이미지경로)
# cv2.resize(이미지,(직접 지정할 크기), fx=현재 이미지 가로의 몇배, fy, = 현재 이미지 세로의 몇배, 보간법)
# cv2.cvtColor(이미지, 색상변환코드)
# cv2.INTER_AREA : 영역 보간법 - when : 사이즈를 줄일 때
# cv2.Canny(이미지,minVal,  , apeture_size)
# np.linspace(시작 , 중지 , num = 50 , 끝점 = True , retstep = False , dtype = None , 축 = 0 )
# np.linspace -> 지정된 간격 동안 균등 한 간격의 숫자를 반환
# cv2.matchTemplate () -> 템플릿 매칭은 원본 이미지에서 특정 이미지를 찾는 방법
# cv2.minMaxLoc () -> 최대/최소값과 그 위치 얻을 수 있음


import cv2
import numpy as np

# 그 후 이미지 크기를 조정하고 종횡비를 유지합니다.
# 너비와 높이의 비율 유지
def  maintain_aspect_ratio_resize(image,  width=None,  height=None,  inter=cv2.INTER_AREA):

#      그런 다음 이미지 크기를 잡고 치수를 초기화합니다.
     dim =  None
     (h, w)  = image.shape[:2]
#      크기를 조정할 필요가 없으면 원본 이미지를 반환합니다.
     if width is  None  and height is  None:
        return image
#      너비가 없으면 높이를 조정합니다.
     if width is  None:
        r = height /  float(h)
        dim =  (int(w * r), height)
#      높이가없는 경우 너비를 조정합니다.
     else:
          r = width /  float(w)
          dim =  (width,  int(h * r))
#      크기가 조정 된 이미지 반환
     return cv2.resize(image, dim,  interpolation=inter)



# 템플릿로드, 그레이 스케일로 변환, 캐니 에지 감지 수행
template = cv2.imread('./project/data/test5.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template,  50,  200)
(tH, tW)  = template.shape[:2]
cv2.imshow("template", template)


# 원본 이미지로드, 회색조로 변환
original_image = cv2.imread('./project/data/test5.jpg')
final = original_image.copy()
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
found =  None


# 더 나은 템플릿 일치를 위해 동적으로 이미지 크기 조정
for scale in np.linspace(0.2,  1.0,  20)[::-1]:
#      [::-1]:처음부터 끝까지 -1칸 간격으로 ( == 역순으로)
     resized = maintain_aspect_ratio_resize(gray,  width=int(gray.shape[1]  * scale))# 원본(그레이스케일) 가로 줄여서 리사이즈
     r = gray.shape[1]  /  float(resized.shape[1]) #원본가로/리사이즈가로(사이즈가 몇배 되었는지)
           
     if resized.shape[0]  < tH or resized.shape[1]  < tW:
        break #조정한 원본 이미지 > temple 이미지 -> break 
     canny = cv2.Canny(resized,  50,  200) #리사이즈 canny
     detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF) #temple이미지 원본에서 찾기
     (_, max_val, _, max_loc)  = cv2.minMaxLoc(detected) #찾은 최댓값(모퉁이)

     if found is  None  or max_val > found[0]:# 클수록 좋다
        found =  (max_val, max_loc, r)


# 경계 상자의 좌표 계산
(_, max_loc, r)  = found
(start_x, start_y)  =  (int(max_loc[0]  * r),  int(max_loc[1]  * r))#시작점*사이즈가 몇배 resize?
(end_x, end_y)  =  (int((max_loc[0]  + tW)  * r),  int((max_loc[1]  + tH)  * r))


# 제거 할 ROI에 경계 상자 그리기
cv2.rectangle(original_image,  (start_x, start_y),  (end_x, end_y),  (0,255,0),  2)
cv2.imshow('detected', original_image)


# 원하지 않는 ROI 지우기 (ROI를 흰색으로 채우기)
cv2.rectangle(final,  (start_x, start_y),  (end_x, end_y),  (255,255,255),  -1)
cv2.imwrite('./project/data/final.jpg', final)
cv2.waitKey(0)

''' 
[1.         0.95789474 0.91578947 0.87368421 0.83157895 0.78947368
 0.74736842 0.70526316 0.66315789 0.62105263 0.57894737 0.53684211
 0.49473684 0.45263158 0.41052632 0.36842105 0.32631579 0.28421053
 0.24210526 0.2       ] '''