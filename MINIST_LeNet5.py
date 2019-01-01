import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

CONV1_SIZE=5
CONV1_DEEP=32

CONV2_SIZE=5
CONV2_DEEP=64

#全连接层节点个数
FC_SIZE=512

#有2个卷积层、2个池化层、2个全连接层
def inference(input_tensor ,train ,regularizer=None):
    #第一层：卷积层
    #输入：28*28*1 ，输出28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weight=tf.get_variable(name=='weight' ,shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        #偏置为每一个channel一个，同一个channel用的是一样的偏置
        conv1_bias=tf.get_variable(name='bias' ,shape=[CONV1_DEEP] ,initializer=tf.constant_initializer(0.0))
        #卷积操作
        conv1=tf.nn.conv2d(input_tensor,conv1_weight,strides=[1,1,1,1],padding="SAME")
        #激活
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))

    #第二层：池化
    #输入：28*28*32 ，输出：14*14*32
    with tf.variable_scope("layer2-pool1"):
        #ksize=过滤器尺寸，strides=步长
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


    #第三层：卷积层
    #输入：14*14*32 ，输出：14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight=tf.get_variable(name=='weight' ,shape=[CONV2_SIZE,CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        #偏置为每一个channel一个，同一个channel用的是一样的偏置
        conv2_bias=tf.get_variable(name='bias' ,shape=[CONV2_DEEP] ,initializer=tf.constant_initializer(0.0))
        #卷积操作
        conv2=tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding="SAME")
        #激活
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))

    #第四层：池化
    #输入：14*14*64 ，输出：7*7*64
    with tf.variable_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1] ,strides=[1,2,2,1],padding=['SAME'])

    #第五层：全连接层
    #先将上一层输出的三维矩阵转换为向量：
    #实际上每一层的输入输出都为一整个batch的矩阵，数据结构：[batch，length，width，channel]
    pool_shape=pool2.get_shape().as_list()
    #计算节点个数：7*7*64
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #pool_shape[0]为一个batch中样本个数，一下转换为一个二维向量，第一个表示第几个样本，第二个表示一个样本的向量
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    #开始进行全连接计算

    with tf.variable_scope('layer5-fc1'):
        #FC_SiZE=512
        fc1_weight=tf.get_variable(name='weight' , shape=[nodes,FC_SIZE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #正则化
        if regularizer!=None:
            tf.add_to_collection("losses" ,regularizer(fc1_weight))

        #偏置
        fc1_bias=tf.get_variable(name='bias' ,shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        #激活
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weight)+fc1_bias)
        #dropout
        if train:
            #keep_prob=drop率
            fc1=tf.nn.dropout(x=fc1,keep_prob=0.5)

    #第六层：全连接层
    with tf.variable_scope("layer6-fc2"):
        fc2_weight=tf.get_variable(name='weight' ,shape=[FC_SIZE,NUM_LABELS],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #正则化，避免过拟合
        if regularizer!=None:
            tf.add_to_collection('lossed' ,regularizer(fc2_weight))

        fc2_bias=tf.get_variable(name='bias' ,shape=[NUM_LABELS] ,initializer=tf.constant_initializer(0.1))
        #激活
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weight)+fc2_weight)
        
    return fc2




