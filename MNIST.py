#-*-coding=utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data



def test_1():
    #读取MNIST数据 没有的话会自动下载到指定的目录
    'one_hot为one_hot编码，即独热码，作用是将状态值编码成状态向量，例如，数字状态共有0~9这10种，'
    '对于数字7，将它进行one_hot编码后为[0 0 0 0 0 0 0 1 0 0]，这样使得状态对于计算机来说更加明确，对于矩阵操作也更加高效。'
    mnist = input_data.read_data_sets('./MNIST_DATA/', one_hot=True)

    #60000个训练集
    print('Training data size :' ,mnist.train.num_examples)
    print('validating data size :' ,mnist.validation.num_examples)
    #10000个测试集
    print('testing data size :' , mnist.test.num_examples)

    print("example training data :" ,mnist.train.images[0])
    print("example training data lable:" , mnist.train.labels[0])

def test_2():
    #随机梯度下降，随机抽取数据进行梯度下降
    mnist = input_data.read_data_sets('./MNIST_DATA/', one_hot=True)
    batch_size=100
    xs,ys=mnist.train.next_batch(batch_size)
    print('x shape' ,xs.shape)

    print('y shape' ,ys.shape)




if __name__=='__main__':
    #test_1()
    test_2()