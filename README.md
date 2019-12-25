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

## 任务二

扩大数据集，使用任务一的网络识别三种不同扰动。
1. 三种扰动信号构建数据集：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP2/wave_dataset-Copy1.ipynb)
2. 一维CNN分类：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP2/wave_classify-Copy1.ipynb)
  >训练集loss为0.006494189302126567，准确率为0.9989984035491943\
   验证集loss为0.08075833001307078，准确率为0.9754464030265808\
   测试集loss为0.08794098829259415，准确率为0.9640718698501587
   
## 任务三

将每条数据长度局限在一个传感器范围内，即50个数据点。
- 构建数据集：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP3/wave_dataset.ipynb)
- 模型的训练与测试：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP3/wave_classify.ipynb)
### 一维信号分类结果

- 4995条训练数据，501条验证数据，501条测试数据
- 训练集准确率为0.9453125, 测试集准确率为0.9061876535415649

### 二维信号分类结果

- 4869条训练数据，489条验证数据，489条测试数据
- 训练集准确率为1.0, 测试集准确率为1.0


## GPR信号识别
