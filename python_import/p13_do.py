import p11_car
import p12_tv

# p11_car.py의 module 이름은 :  p11_car
                        # -> p11_car.py에서는 __main__
                        # 불러온 애의 파일명이 출력


print('======================================')
print('p13_do.py의 modeule 이름은 : ', __name__)    
# ---->p13_do.py의 modeule 이름은 :  __main__
print('======================================')

p11_car.drive()
p12_tv.watch()
# 운전하다
# 시청하다