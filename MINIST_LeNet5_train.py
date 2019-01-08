#-*- coding: utf-8 -*-

import MINIST_LeNet5_inference as inference
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import  input_data

# 1.定义神经网络相关的参数
BATCH_SIZE=100
#基础学习率
LEARNING_RATE_BASE=0.1
#学习衰减率
LEARNING_RATE_DECAY=0.99
#正则化比率
REGULARIZATION_RATE=0.0001
#训练轮数
TRAINING_STEPS=6000
#滑动平均衰减率
MOVING_AVERAGE_DECAY=0.99

# 2.定义训练过程
def train(mnist):
    #定义输入的placeholder
    x=tf.placeholder(dtype=tf.float32 ,shape=[BATCH_SIZE,inference.IMAGE_SIZE,inference.IMAGE_SIZE,inference.NUM_CHANNELS],name="x-input")
    #y_真实值
    y_=tf.placeholder(tf.float32,[None, inference.OUTPUT_NODE],name='y-input')
    #正则化
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y=inference.inference(x,False,regularizer)

    global_step = tf.Variable(0,trainable=False)
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))



def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()








