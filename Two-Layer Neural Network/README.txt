HW4: 两层神经网络

1。项目简介
实现一个两层全连接神经网络，包含手动反向传播和线性搜索学习率。

2.运行方法
（1）安装依赖
在终端输入：pip install numpy torch matplotlib
（2）运行代码 
在终端输入：python main.py

3.文件说明
（1）main.py - 主训练程序
（2）utils.py - 工具函数
（3）homework_features_256_20000.pth - 数据集
（4）loss_curve.png - 生成的损失曲线图

4.输出文件
运行后会生成：
（1）loss_curve.png - 训练损失曲线
（2）model_parameters.npz - 训练好的模型参数

5.注意事项
（1）确保数据文件在同一目录下
（2）程序会自动处理OpenMP库冲突
（3）训练过程会显示进度和最终结果