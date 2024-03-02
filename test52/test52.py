# K均值(K-Means)聚类
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# OpenCV提供了 cv2.kmeans()函数实现K-Means聚类算法，该算法找到簇的中心并将输入样本分组到簇周围。
# K-Means聚类算法的目标是将 n 个样本划分(聚类)为 K 个簇，其中每个样本都属于具有最近均值的簇，
# cv2.kmeans()函数用法如下：
# retval, bestLabels, centers=cv2.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
# data 表示用于聚类的输入数据，它是 np.float32 数据类型，每一列包含一个特征；
# K 指定最后需要的簇数；
# 算法终止标准由 criteria 参数指定，该参数设置最大迭代次数或所需精度，当满足这些标准时，算法终止。
# criteria 是具有三个参数 (type, max_item, epsilon) 的元组：（见截屏图片）
# attempts 参数指定使用不同的初始标签执行算法的次数。
# flags 参数指定初始化簇中心的方法，其可选值包括：
#     cv2.KMEANS_RANDOM_CENTERS 每次选择随机初始化簇中心；
#     cv2.KMEANS_PP_CENTERS 使用 Arthur等人提出的 K-Means++中心初始化。

# cv2.kmeans() 返回以下内容：（见截屏图片）