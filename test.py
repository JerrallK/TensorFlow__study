# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState


def test_1():
    with tf.Session():
        input1 = tf.constant([1.0, 1.0, 1.0, 1.0])
        input2 = tf.constant([2., 3., 4., 8.])
        input3 = tf.constant([2., 6., 4., 8.])
        output = tf.add(input1, input2, name="ee")
        output1 = tf.add(input1, input3, name="ee")
        print("result:", output.eval())
        print("result:", output1)
    print([x + y for x, y in zip([1.0] * 4, [2.0] * 4)])


def test_2():
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    result = a + b
    print(a.graph is tf.get_default_graph())


def test_3():
    g1 = tf.Graph()
    with g1.as_default():
        v = tf.get_variable("v", shape=[2, 2], initializer=tf.zeros_initializer())
    with tf.Session(graph=g1) as session1:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print(session1.run(tf.get_variable("v")))
    g2 = tf.Graph()
    with g2.as_default():
        v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())
    with tf.Session(graph=g2) as session1:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print(session1.run(tf.get_variable("v")))


def test_4():
    w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
    x = tf.constant([])
    with tf.Session() as sess:
        # 需要初始化！
        sess.run(w1.initializer)
        print("sdf", w1.eval())


def test5():
    batch_size = 8
    w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name="w1")
    w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name='w2')
    # 定义placeholder
    # 作为存放输入数据的地方。这里维度也不一定要定义。
    # 但如果维度是确定的，那么给出维度可以降低出错的概率。
    # x=tf.constant([[0.7,0.9]],name='x')

    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
    y_ = tf.placeholder(tf.float32, shape=(None, 2), name="y-input")

    a = tf.matmul(x, w1, name='a')
    y = tf.matmul(a, w2, name='y')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.2, 0.6], [0.9, 0.5]]}))
        writer = tf.summary.FileWriter("F:\PycharmProjects\Tensorflow\log666", tf.get_default_graph())
        writer.close()

        y = tf.sigmoid(y)
        cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                        + (1 - y) * tf.log(tf.clip_by_value(-y, 1e-10, 1.0)))

        learnint_rate = 0.001
        train_step = tf.train.AdamOptimizer(learnint_rate).minimize(cross_entropy)

        rdm = RandomState()
        dataset_size = 128

        X = rdm.rand(dataset_size, 2)
        Y = [[int(xl + x2 < 1)] for (xl, x2) in X]


def test6():
    # 自定义损失函数

    # 定义神经网络的相关参数和变量
    batch_size = 8
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
    # y_: 真实值
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    # y: 预测值
    y = tf.matmul(x, w1)

    # 设置自定义的损失函数
    loss_less = 10
    loss_more = 1
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 生成模拟数据集(假设这些都是真实数据 因为没有真实数据来源 所以只能生成)
    rdm = RandomState(1)  # seed:1
    X = rdm.rand(128, 2)  # shape : 128,2
    Y = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

    # 训练模型
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 5000
        for i in range(STEPS):
            start = (i * batch_size) % 128
            end = (i * batch_size) % 128 + batch_size
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 == 0:
                print('after %d training steps,w1 is :' % i)
                print(sess.run(w1), '\n')
        print('final w1 =', '\n', sess.run(w1))
        writer = tf.summary.FileWriter("/home/jerrall/tensorflow_study/tensorboard", tf.get_default_graph())
        writer.close()

def test_7():
    #假设loss函数 y=x^2, 选择初始点   x_0=5
    Training_step=100
    #Learning_rate=0.01
    global_step = tf.Variable(0)#计数器，每训练一个batch就+1
    Learning_rate=tf.train.exponential_decay(0.1,global_step,1,0.96,staircase=True)
    x=tf.Variable(tf.constant(5,dtype=tf.float32),name='x_0')
    y=tf.square(x)

    train_op=tf.train.GradientDescentOptimizer(Learning_rate).minimize(y,global_step=global_step)

    with tf.Session() as  sess:
        sess.run(tf.global_variables_initializer())
        for i in range(Training_step):
            sess.run(train_op)
            x_value=sess.run(x)
            print ("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value) )



if __name__ == "__main__":
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    #test6()
    test_7()
