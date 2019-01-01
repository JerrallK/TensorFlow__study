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

#ȫ���Ӳ�ڵ����
FC_SIZE=512

#��2������㡢2���ػ��㡢2��ȫ���Ӳ�
def inference(input_tensor ,train ,regularizer=None):
    #��һ�㣺�����
    #���룺28*28*1 �����28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weight=tf.get_variable(name=='weight' ,shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        #ƫ��Ϊÿһ��channelһ����ͬһ��channel�õ���һ����ƫ��
        conv1_bias=tf.get_variable(name='bias' ,shape=[CONV1_DEEP] ,initializer=tf.constant_initializer(0.0))
        #�������
        conv1=tf.nn.conv2d(input_tensor,conv1_weight,strides=[1,1,1,1],padding="SAME")
        #����
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))

    #�ڶ��㣺�ػ�
    #���룺28*28*32 �������14*14*32
    with tf.variable_scope("layer2-pool1"):
        #ksize=�������ߴ磬strides=����
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


    #�����㣺�����
    #���룺14*14*32 �������14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight=tf.get_variable(name=='weight' ,shape=[CONV2_SIZE,CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        #ƫ��Ϊÿһ��channelһ����ͬһ��channel�õ���һ����ƫ��
        conv2_bias=tf.get_variable(name='bias' ,shape=[CONV2_DEEP] ,initializer=tf.constant_initializer(0.0))
        #�������
        conv2=tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding="SAME")
        #����
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))

    #���Ĳ㣺�ػ�
    #���룺14*14*64 �������7*7*64
    with tf.variable_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1] ,strides=[1,2,2,1],padding=['SAME'])

    #����㣺ȫ���Ӳ�
    #�Ƚ���һ���������ά����ת��Ϊ������
    #ʵ����ÿһ������������Ϊһ����batch�ľ������ݽṹ��[batch��length��width��channel]
    pool_shape=pool2.get_shape().as_list()
    #����ڵ������7*7*64
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #pool_shape[0]Ϊһ��batch������������һ��ת��Ϊһ����ά��������һ����ʾ�ڼ����������ڶ�����ʾһ������������
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    #��ʼ����ȫ���Ӽ���

    with tf.variable_scope('layer5-fc1'):
        #FC_SiZE=512
        fc1_weight=tf.get_variable(name='weight' , shape=[nodes,FC_SIZE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #����
        if regularizer!=None:
            tf.add_to_collection("losses" ,regularizer(fc1_weight))

        #ƫ��
        fc1_bias=tf.get_variable(name='bias' ,shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        #����
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weight)+fc1_bias)
        #dropout
        if train:
            #keep_prob=drop��
            fc1=tf.nn.dropout(x=fc1,keep_prob=0.5)

    #�����㣺ȫ���Ӳ�
    with tf.variable_scope("layer6-fc2"):
        fc2_weight=tf.get_variable(name='weight' ,shape=[FC_SIZE,NUM_LABELS],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #���򻯣���������
        if regularizer!=None:
            tf.add_to_collection('lossed' ,regularizer(fc2_weight))

        fc2_bias=tf.get_variable(name='bias' ,shape=[NUM_LABELS] ,initializer=tf.constant_initializer(0.1))
        #����
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weight)+fc2_weight)
        
    return fc2




