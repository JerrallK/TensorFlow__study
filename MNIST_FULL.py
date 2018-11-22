#-*-coding=utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

#完整的MNIST案例：

INPUT_NODE=784                     #输入节点784个，代表784个像素
OUTPUT_NODE=10                     #输出节点10个，代表10中不同的分类情况

LAYER1_NODE=500                    #只有一个隐藏层，但有500个节点

BATCH_SIZE=100                     #一个batch的个数，数字越大越接近梯度下降，越小越接近随机梯度下降

LEARNING_RATE_BASE=0.8             #基础学习率

LEARNING_RATE_DECAY=0.99           # 学习率的衰减率decayed_learning_rate = \
                                   # learning_rate * decay_rate^(global_step/decay_ steps)
                                   # decay_ steps通常是学习的迭代轮数

REGULARIZATION_RATE=0.0001         #正则化项的系数

TRAINING_STEPS=30000               #训练轮数

MOVING_AVERAGE_DECAY=0.99          # 滑动平均衰减率

#计算向前传播的
def inference(input_tensor, avg_class, weights1,biases1,weights2,biases2):
    if avg_class==None:

        #仅有一个影藏层
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

        #由于最后输出就直接经过处理变为概率，就不需要再一次进行激活了
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1=tr.nn.relu(tf.matmul(input_tensor,weights1)+biases1)+avg_class.average(biases1)

        return tf.matmul(tf.matmul(layer1,weights2)+biases2)+avg_class.average(biases2)

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #随机生成影藏层的参数，
    #tf.truncated_normal是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成输出层的参数:
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算向前传播结果
    y=inference(x,None,weights1,biases1,weights2,biases2)

    #
    global_step=tf.Variable(0,trainable=False)





if __name__=='__main__':
    #test_1()
    test_2()