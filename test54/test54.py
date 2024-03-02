# 支持向量机
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 支持向量机（SVM）是一种监督学习技术，它通过根据指定的类对训练数据进行最佳分离，从而在高维空间中构建一个或一组超平面。
# 以二维平面为例：在参考图中可以看到，其中绿线是能够将两个类分开的最佳超平面，因为其到两个类中的最近元素的距离是最大的。

# OpenCV中的SVM实现基于LIBSVM，使用cv2.ml.SVM_create()函数创建空模型，然后为模型分配主要参数：
# svmType ：设置SVM类型，可选值如下：
#     SVM_C_SVC：C-支持向量分类，可用于 n 分类(n≥2)问题
#     NU_SVC: v-支持向量分类
#     ONE_CLASS: 分布估计(单类 SVM)
#     EPS_SVR: ϵ-支持向量回归
#     NU_SVR: v-支持向量回归

# kernelType ：这设置了 SVM 的核类型，可选值如下：
#     LINEAR : 线性核
#     POLY ：多项式核
#     RBF : Radial Basis Function (RBF)，大多数情况下是不错的选择
#     SIGMOID : Sigmoid 核
#     CHI2 : 指数 Chi2 核，类似于 RBF 核
#     INTER : 直方图交集核；运行速度较快的核

# degree : 核函数的 degree 参数 (用于 POLY 核)
# gamma ：核函数的 γ 参数（用于 POLY/RBF/SIGMOID/CHI2 核）
# coef0 : 核函数的 coef0 参数 (用于 POLY/SIGMOID 核)

# Cvalue : SVM 优化问题的 C 参数 (用于 C_SVC/EPS_SVR/NU_SVR 类型)
# nu : SVM 优化问题的 v 参数 (用于 NU_SVC/ONE_CLASS/NU_SVR 类型)
# p : SVM 优化问题的 ϵ 参数 (用于 EPS_SVR 类型)

# classWeights : C_SVC 问题中的可选权重，分配给特定的类
# termCrit ：迭代 SVM 训练过程的终止标准

# 核函数选择通常取决于数据集，通常可以首先使用 RBF 核进行测试，因为该核将样本非线性地映射到更高维空间，可以方便的处理类标签和属性之间的关系是非线性的情况。
# 默认构造函数使用以下值初始化 SVM：
# svmType: C_SVC, kernelType: RBF, degree: 0, gamma: 1, coef0: 0, C: 1, nu: 0, p: 0, classWeights: 0, termCrit: TermCriteria(MAX_ITER+EPS, 1000, FLT_EPSILON )
