# [실습]
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)


node_add = tf.add(node1, node2)
node_sub = tf.subtract(node1, node2)
node_mul = tf.multiply(node1, node2)
node_div = tf.divide(node1, node2)          #x의해 파이썬 스타일 나누기를 계산
node_truediv = tf.truediv(node1, node2)     #x / y를 요소별로 나눕니다
node_floordiv = tf.floordiv(node1, node2)   #x / y가장 음의 정수로 반올림
node_mod = tf.mod(node1, node2)

sess = tf.Session()
print('sess.run(node1, node2) : ', sess.run([node1, node2]))  #sess.run(node1, node2) :  [2.0, 3.0]
print('sess.run(node_add) : ', sess.run(node_add))   #sess.run(node_add) :  5.0
print('sess.run(node_sub) : ', sess.run(node_sub))   #sess.run(node_sub) :  -1.0
print('sess.run(node_mul) : ', sess.run(node_mul))   #sess.run(node_mul) :  6.0
print('sess.run(node_div) : ', sess.run(node_div))   #sess.run(node_div) :  0.6666667
print('sess.run(node_truediv) : ', sess.run(node_truediv))      #sess.run(node_truediv) :  0.6666667
print('sess.run(node_floordiv) : ', sess.run(node_floordiv))    #sess.run(node_floordiv) :  0.0
print('sess.run(node_mod) : ', sess.run(node_mod))   #sess.run(node_mod) :  2.0



# TensorFlow 연산   축약 연산자   설명
# tf.add()   a + b   a와 b를 더함
# tf.multiply()   a * b   a와 b를 곱함
# tf.subtract()   a - b   a에서 b를 뺌
# tf.divide()   a / b   a를 b로 나눔
# tf.pow()   a ** b     를 계산
# tf.mod()   a % b   a를 b로 나눈 나머지를 구함
# tf.logical_and()   a & b   a와 b의 논리곱을 구함. dtype은 반드시 tf.bool이어야 함
# tf.greater()   a > b     의 True/False 값을 반환
# tf.greater_equal()   a >= b     의 True/False 값을 반환
# tf.less_equal()   a <= b     의 True/False 값을 반환
# tf.less()   a < b     의 True/False 값을 반환
# tf.negative()   -a   a의 반대 부호 값을 반환
# tf.logical_not()   ~a   a의 반대의 참거짓을 반환. tf.bool 텐서만 적용 가능
# tf.abs()   abs(a)   a의 각 원소의 절대값을 반환
# tf.logical_or()   a I b   a와 b의 논리합을 구함. dtype은 반드시 tf.bool이어야 함
