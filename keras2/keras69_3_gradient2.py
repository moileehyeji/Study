# keras69_2_gradient1의 (2,2)지점 찾기

import numpy as np

f = lambda x: x**2 - 4*x + 6

gradient = lambda x: 2*x - 4    #f 미분

x0 = 20     #임의의 가중치
epoch = 100
learning_rate = 0.01

print('step\tx\tf(x)')
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)    #10 - 0.1*16 = 8.4(가중치)     
    x0 = temp

    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))

#경사하강법 : x - learning rate * 미분식(x)
# -> 가중치(x)를 (learning rate * 미분식(x))만큼 줄여나가면서 최적의 가중치를 찾는 루프
# 가중치(x)증가     -> epoch증가
# learnin rate감소  -> epoch증가
