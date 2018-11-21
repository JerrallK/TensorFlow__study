#-*-coding=utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data



def test_1():
    #读取MNIST数据 没有的话会自动下载到指定的目录
    'one_hot为one_hot编码，即独热码，作用是将状态值编码成状态向量，例如，数字状态共有0~9这10种，'
    '对于数字7，将它进行one_hot编码后为[0 0 0 0 0 0 0 1 0 0]，这样使得状态对于计算机来说更加明确，对于矩阵操作也更加高效。'
    mnist = input_data.read_data_sets('./MNIST_DATA/', one_hot=True)
    print('Training data size :' ,mnist.train.num_examples)
    print('validating data size :' ,mnist.validation.num_examples)
    print('testing data size :' , mnist.test.num_examples)
    print("example training data :" ,mnist.train.images[0])
    print("example training data lable:" , mnist.train.labels[0])


if __name__=='__main__':
    test_1()