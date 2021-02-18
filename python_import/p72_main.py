import p71_byunsu as p71 

# p71.aaa = 3   #print(p71.aaa) 3출력

print(p71.aaa)
print(p71.square(10))
# 2
# 1024

print('===================================')

from p71_byunsu import aaa, square

print(aaa)  #p71의 aaa 메모리
# 2

aaa = 3

print(aaa)  #p72의 aaa 메모리
print(square(10))   #p71의 square함수 메모리
# 3
# 1024

# p71의 aaa(=2)변수, square함수가 메모리에 할당
# p72의 aaa(=3)변수가 다른 메모리에 할당