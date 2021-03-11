#accuracy_score 써보자

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 

#[실습]
# accuracy_score로 결론 낼것


#input
x_data, y_data = load_iris(return_X_y=True)
y_data = y_data.reshape(-1,1)
# print(x_data.shape, y_data.shape) #(150, 4) (150, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, shuffle = True, random_state = 66)

# 전처리
onehot = OneHotEncoder()
onehot.fit(y_data)
y_train = onehot.transform(y_train).toarray()

sc = MinMaxScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


#input
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4,3]), name = 'weight')
b = tf.Variable(tf.random_normal([1,3]), name = 'bias')

# hypothesis = [None,3]
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

# train = tf.train.GradientDescentOptimizer(learning_rate = 1e-6).minimize(cost) 
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

# prediction = tf.argmax(hypothesis, 1)
# correct_prediction  =  tf.cast(hypothesis > 0.5, dtype = tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, y), dtype = tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step  in range(10001):
        cost_val, _= sess.run([cost, train], feed_dict = {x:x_train, y:y_train})

        if step % 1000 == 0:
            print(step, cost_val)

    #predict
    # hy_val, acc = sess.run([hypothesis, accuracy],feed_dict = {x:x_test, y:y_test} )
    # print(sess.run(tf.argmax(hy_val, axis = -1))[:5], acc)  #[1 1 1 0 1] 0.95555556
    pred = sess.run(hypothesis, feed_dict = {x:x_test})
    pred = sess.run(tf.argmax(pred,1))
    print('pred : ', pred)

    #accuracy_score
    acc = accuracy_score(pred, y_test) 
    print('acc : ', acc)    #acc :  0.9666666666666667
