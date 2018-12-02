# -*-coding=utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState
import time
from tensorflow.examples.tutorials.mnist import input_data

# 完整的MNIST案例：

INPUT_NODE = 784  # 输入节点784个，代表784个像素
OUTPUT_NODE = 10  # 输出节点10个，代表10中不同的分类情况

LAYER1_NODE = 500  # 只有一个隐藏层，但有500个节点

BATCH_SIZE = 100  # 一个batch的个数，数字越大越接近梯度下降，越小越接近随机梯度下降

LEARNING_RATE_BASE = 0.8  # 基础学习率

LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率decayed_learning_rate = \
# learning_rate * decay_rate^(global_step/decay_ steps)
# decay_ steps通常是学习的迭代轮数

REGULARIZATION_RATE = 0.0001  # 正则化项的系数

TRAINING_STEPS = 30000  # 训练轮数

MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 计算向前传播的
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:

        # 仅有一个影藏层
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 由于最后输出就直接经过处理变为概率，就不需要再一次进行激活了
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)) + avg_class.average(
            biases1)

        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 随机生成隐藏层的参数，
    # tf.truncated_normal是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数:
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算向前传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    #
    global_step = tf.Variable(0, trainable=False)

    variable_averagers = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averagers_op = variable_averagers.apply(tf.trainable_variables())

    average_y = inference(x, variable_averagers, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # train_step or train_operation
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 让with里的操作都在参数的环境里执行，以下操作等价于
    # train_op=tf.group(train_step,variable_averagers_op),创建一个包含所有input的操作
    with tf.control_dependencies([train_step, variable_averagers_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 转换数据类型，bool中的true为1，false为0
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # config=tf.ConfigProto(device_count={"CPU": 12}, inter_op_parallelism_threads=0,intra_op_parallelism_threads=0)
    with tf.Session(config=tf.ConfigProto(device_count={"CPU": 12}, inter_op_parallelism_threads=0,
                                          intra_op_parallelism_threads=0)) as sess:
        tf.global_variables_initializer().run()

        # 验证集，用于判断训练大致的停止条件和评判训练的结果
        validata_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 测试集，用于训练结束后评判模型的优劣
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        start = time.time()
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validata_acc = sess.run(accuracy, feed_dict=validata_feed)

                print("After %d training step(s) validation accuracy using average model is %g " % (i, validata_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        end = time.time()
        print("time cost :%g" % (end - start))
        print("After %d training step(s) ,test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
