# 神经网络的学习与实践
<!-- TOC -->

- [神经网络的学习与实践](#%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e7%9a%84%e5%ad%a6%e4%b9%a0%e4%b8%8e%e5%ae%9e%e8%b7%b5)
  - [基础学习](#%e5%9f%ba%e7%a1%80%e5%ad%a6%e4%b9%a0)
    - [MNIST DATASET](#mnist-dataset)
      - [code for a 3-layer neural network using numpy](#code-for-a-3-layer-neural-network-using-numpy)
      - [code for a Softmax Regression using Tensorflow](#code-for-a-softmax-regression-using-tensorflow)
      - [CNN](#cnn)
        - [Based on Tensorflow framework](#based-on-tensorflow-framework)
        - [Based on Keras](#based-on-keras)
    - [learning PyTorch](#learning-pytorch)
  - [一维信号分类任务](#%e4%b8%80%e7%bb%b4%e4%bf%a1%e5%8f%b7%e5%88%86%e7%b1%bb%e4%bb%bb%e5%8a%a1)
    - [任务一](#%e4%bb%bb%e5%8a%a1%e4%b8%80)
  - [任务二](#%e4%bb%bb%e5%8a%a1%e4%ba%8c)
  - [任务三](#%e4%bb%bb%e5%8a%a1%e4%b8%89)
    - [基于联合采集数据集](#%e5%9f%ba%e4%ba%8e%e8%81%94%e5%90%88%e9%87%87%e9%9b%86%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [基于混合数据集](#%e5%9f%ba%e4%ba%8e%e6%b7%b7%e5%90%88%e6%95%b0%e6%8d%ae%e9%9b%86)
  - [基于剔除无效信号的混合数据集](#%e5%9f%ba%e4%ba%8e%e5%89%94%e9%99%a4%e6%97%a0%e6%95%88%e4%bf%a1%e5%8f%b7%e7%9a%84%e6%b7%b7%e5%90%88%e6%95%b0%e6%8d%ae%e9%9b%86)
  - [GPR信号识别](#gpr%e4%bf%a1%e5%8f%b7%e8%af%86%e5%88%ab)

<!-- /TOC -->
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
1. 原始数据：四种扰动信号 --> 构建数据集: [code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP1/wave_dataset.ipynb)
2. 搭建一维CNN进行分类: [code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP1/wave_classify.ipynb)
3. 训练集准确率为99.7%, 测试集准确率为100%

## 任务二

扩大数据集，使用任务一的网络识别三种不同扰动。
1. 三种扰动信号构建数据集：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP2/wave_dataset-Copy1.ipynb)
2. 一维CNN分类：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP2/wave_classify-Copy1.ipynb)
3. 训练集准确率为99.9%, 测试集准确率为96.4%
   
## 任务三

将每条数据长度局限在一个传感器范围内，即50个数据点。
- 构建数据集：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP3/wave_dataset.ipynb)
- 模型的训练与测试：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP3/wave_classify.ipynb)

**以下均采用同一个一维CNN模型**
### 基于联合采集数据集
1. 一维信号分类结果
   - 4995条训练数据，501条验证数据，501条测试数据
   - 训练集准确率为94.5%, 测试集准确率为90.6%

2. 二维信号分类结果
   - 4869条训练数据，489条验证数据，489条测试数据 
   - 训练集准确率为100%, 测试集准确率为100%

### 基于混合数据集
1. 一维混合信号分类结果
   - 19992条训练数据，1998条验证数据，1998条测试数据
   - 训练集准确率为76.8%, 测试集准确率为70.7%
2. 二维混合信号分类结果
   - 19866条训练数据，1986条验证数据，1986条测试数据
   - 训练集准确率为100%, 测试集准确率为100%

## 基于剔除无效信号的混合数据集
1. 一维有效信号分类结果
   - 7329条训练数据，732条验证数据，732条测试数据
   - 训练集准确率为91.2%, 测试集准确率为72.4%
2. 二维有效信号分类结果
   - 12198条训练数据，1219条验证数据，1219条测试数据
   - 训练集准确率为99.9%, 测试集准确率为100%

## GPR信号识别
