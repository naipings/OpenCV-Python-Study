# K最近邻示例
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 接下来，为了演示kNN算法，首先随机创建一组点并分配一个标签（0或1）。
# 标签0将代表红色三角形，而标签1将代表蓝色三角形；
# 然后，使用KNN算法根据 K 个最近邻对样本点进行分类。

# 第一步是创建具有相应标签的点集和要分类的样本点：
# 点集由50个点组成
data = np.random.randint(0, 100, (50, 2)).astype(np.float32)
# 为1每个点创建标签 (0:红色, 1:蓝色)
labels = np.random.randint(0, 2, (50, 1)).astype(np.float32)
# 创建要分类的样本点
sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# 接下来，创建kNN分类器，训练分类器，并找到要分类样本点的 K 个最近邻居：
# 创建 kNN 分类器
knn = cv2.ml.KNearest_create()
# 训练 kNN 分类器
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
# 找到要分类样本点的 k 个最近邻居
k = 3
ret, results, neighbours, dist = knn.findNearest(sample, k)
# 打印结果
print("result: {}".format(results))
print("neighbours: {}".format(neighbours))
print("distance: {}".format(dist))

# 可视化
fig = plt.figure(figsize=(8, 6))
red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')
blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')
plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')
plt.show()

# 控制台将打印结果，根据获取到的结果可知，由于我们设置k=3，所以有：
# 如果离绿点最近的3个点，红点占多数，则绿点被归类为红色三角形，可视化图见运行结果。
# 如果离绿点最近的3个点，蓝点占多数，则绿点被归类为蓝色正方形，可视化图见运行结果。

