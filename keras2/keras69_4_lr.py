x_train = 0.5
y_train = 0.8

weight = 0.8
lr = 0.01
epoch = 5

for iteration in range(epoch):
    y_predict = x_train * weight        #예측값(0.5*0.5=0.25)
    error = (y_predict - y_train) **2   #loss(예측값과 실제값 차이)

    print('Error : ' + str(error) + '\ty_predict : ' + str(y_predict))

    up_y_predict = x_train * (weight + lr)  # x * learning rate만큼 늘린 가중치 
    up_error = (y_train - up_y_predict) **2 #선과의 거리측정(mse)

    down_y_predict = x_train * (weight - lr)    # x * learning rate만큼 줄인 가중치 
    down_error = (y_train - down_y_predict) **2 #선과의 거리측정(mse)

    if(down_error <= up_error): #거리가 멀면
        weight = weight - lr    #가중치 learning rate만큼 줄이기   

    if(down_error > up_error):  #거리가 가까우면
        weight = weight + lr    #가중치 learning rate만큼 늘리기  

# lr이 커지거나 작아지면서 가중치를 찾기위해 왔다갔다
# lr감소 -> epoch증가
# lr증가 -> y_predict간의 격차 증가

