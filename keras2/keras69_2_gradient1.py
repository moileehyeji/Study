import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6   #2차함수
x = np.linspace(-1, 6, 100)
y = f(x)

#시각화
plt.plot(x, y, 'k-')    # 'k':검정색, '-':실선 스타일
plt.plot(2, 2, 'sk')    #(2,2)지점 점이 찍어짐, 's':정사각형 마커
# 마커형식 참고 https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#(2,2)지점 즉, 최적의 가중치를 찾자


