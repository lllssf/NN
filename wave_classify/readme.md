# 扰动信号分类

## 任务一
从分布式光纤同一位置依次制造三种不同扰动，进行识别。数据量较小，四种不同类别的信号每种500组，共2000组数据。
1. 原始数据（四种扰动信号） --> 构建数据集：[code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP1/wave_dataset.ipynb)
2. 搭建一维CNN进行分类: [code](https://github.com/lllssf/NN-implemantation/blob/master/wave_classify/STEP1/wave_classify.ipynb)
  > 训练集loss为0.029167114989832044，准确率为0.9975961446762085\
    验证集loss为0.0305146723985672，准确率为1.0\
    测试集loss为0.033884770986510486，准确率为1.0

## 任务二

