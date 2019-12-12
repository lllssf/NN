# 神经网络的学习与实践
## 基础学习
### MNIST DATASET
#### code for a 3-layer neural network using numpy
- [code](https://github.com/lllssf/NN-implemantation/blob/master/MNIST/3_layers_NN.py)
- Test accuracy: 97.3%
#### code for a Softmax Regression using Tensorflow
- [code](https://github.com/lllssf/NN-implemantation/blob/master/MNIST/sigle_softmax_regression.py) 
- Test accuracy: 92.5%
#### CNN 
##### Based on Tensorflow framework
- [code](https://github.com/lllssf/NN-implemantation/blob/master/MNIST/CNN.py)
- Layers: conv1+pool1+conv2+pool2+fc1+softmax
- Test accuracy: 99.2%
##### Based on Keras
- [code](https://github.com/lllssf/NN-implemantation/blob/master/MNIST/CNN_keras.py)
- Test accuracy: 99.2%
### learning PyTorch
- [基础](https://github.com/lllssf/NN-implemantation/blob/master/torch_tutor.ipynb)

## 一维信号分类任务
主要目标是识别分布式光纤系统上不同的扰动信号，参见[详情](https://github.com/lllssf/NN-implemantation/tree/master/wave_classify)
### 任务一
从分布式光纤同一位置依次制造三种不同扰动，进行识别。
1. 原始数据：四种扰动信号 --> 构建数据集
2. 搭建一维CNN进行分类
3. > 训练集loss为0.029167114989832044，准确率为0.9975961446762085\
验证集loss为0.0305146723985672，准确率为1.0\
测试集loss为0.033884770986510486，准确率为1.0

## GPR信号识别
