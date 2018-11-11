import tensorflow as tf


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
if __name__ == "__main__":
    #test_1()
    #test_2()
    #test_3()
    test_4()