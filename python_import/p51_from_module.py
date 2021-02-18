from machine.car import drive
from machine.tv import watch

drive()
watch()
# 운전하다2
# 시청하다2

print('================================')

# from machine import car
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()
# 운전하다2
# 시청하다2


print('================test================')
from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()
# test 운전하다2
# test 시청하다2

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()
# test 운전하다2
# test 시청하다2

from machine import test

test.car.drive()
test.tv.watch()
# test 운전하다2
# test 시청하다2


# 문제점 :  같은 폴더 내의 모듈만 호출할수 있음
# --> sklearn는 어떻게 호출할 수 있었을까?
# --> Anaconda 설치할때 환경변수 체크한것이 'C:\\Users\\ai\\Anaconda3' 하위 폴더 모든 경로의 모듈을 호출할 수 잇도록 설정한 것