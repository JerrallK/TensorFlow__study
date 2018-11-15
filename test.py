# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState


def test_1():
    with tf.Session():
        input1 = tf.constant([1.0, 1.0, 1.0, 1.0])
        input2 = tf.constant([2., 3., 4., 8.])
        input3 = tf.constant([2., 6., 4., 8.])
        output = tf.add(input1, input2,name="ee")
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
        v = tf.get_variable("v", shape=[2,2],initializer=tf.zeros_initializer())
    with tf.Session(graph=g1) as session1:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(session1.run(tf.get_variable("v")))
    g2=tf.Graph()
    with g2.as_default():
        v = tf.get_variable("v", shape=[1],initializer=tf.zeros_initializer())
    with tf.Session(graph=g2) as session1:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(session1.run(tf.get_variable("v")))
def test_4():
    w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
    x=tf.constant([])
    with tf.Session() as sess:

        #需要初始化！
        sess.run(w1.initializer)
        print("sdf",w1.eval())

def test5():
    batch_size=8
    w1=tf.Variable(tf.random_normal((2,3),stddev=1,seed=1),name="w1")
    w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1),name='w2')
    # 定义placeholder
    # 作为存放输入数据的地方。这里维度也不一定要定义。
    # 但如果维度是确定的，那么给出维度可以降低出错的概率。
    #x=tf.constant([[0.7,0.9]],name='x')

    x=tf.placeholder(tf.float32,shape=(None,2), name="x-input")
    y_=tf.placeholder(tf.float32,shape=(None,2), name="y-input")

    a=tf.matmul(x,w1,name='a')
    y=tf.matmul(a,w2,name='y')

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.2,0.6],[0.9,0.5]]}))
        writer=tf.summary.FileWriter("F:\PycharmProjects\Tensorflow\log666" ,tf.get_default_graph())
        writer.close()


        y=tf.sigmoid(y)
        cross_entropy=-tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
                                      +(1-y)*tf.log(tf.clip_by_value(-y,1e-10,1.0)))

        learnint_rate=0.001
        train_step=tf.train.AdamOptimizer(learnint_rate).minimize(cross_entropy)

        rdm=RandomState()
        dataset_size=128

        X=rdm.rand(dataset_size,2)
        Y = [[int(xl + x2 < 1)] for (xl, x2) in X]
if __name__ == "__main__":
    #test_1()
    #test_2()
    #test_3()
    test_4()