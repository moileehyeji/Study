# import p31_sample 
from p31_sample import test

# import해서 클래스, 함수, 변수 호출 가능

x = 222

def main_func():
    print('x : ', x)

main_func() #x = 222    

# p31_sample.test()   #x :  111
test()  #x :  111

# 두 x의 메모리가 다름