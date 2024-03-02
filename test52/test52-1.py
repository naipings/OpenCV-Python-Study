# K-Means聚类示例
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 作为示例，我们将使用K-Means聚类算法对一组2D点进行聚类。这组2D点由240个点组成，使用两个特征进行了描述：
# 2D数据
data = np.float32(np.vstack((np.random.randint(0, 50, (80, 2)), np.random.randint(40, 90, (80, 2)), np.random.randint(70, 110, (80, 2)))))
# 可视化
# plt.scatter(data[:, 0], data[:, 1], c='c')
# plt.show()
# 执行上面的代码，可以在图中看到，数据将作为聚类算法的输入，每个数据点有两个特征对应于(x, y)坐标，
# 例如，这些坐标可以表示240个人的身高和体重，而K-Means聚类算法用于决定衣服的尺寸(例如K=3，则对应表示尺寸为S、M或L)。

# 接下来，我们将数据划分为 2 个簇：
# 第一步是定义算法终止标准，将最大迭代数设置为20(max_iterm = 20)，epsilon设置为1.0(epsilon = 1.0)：
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

# 然后调用cv2.kmeans()函数应用K-Means算法：
# ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # 由于返回值label存储每个样本的聚类索引，因此，我们可以根据label将数据拆分为不同的集群：
# A = data[label.ravel() == 0]
# B = data[label.ravel() == 1]

# # 最后绘制 A 和 B 以及聚类前后的数据，以便更好地理解聚类过程：
# fig = plt.figure(figsize=(12, 6))
# plt.suptitle("K-means clustering algorithm", fontsize=14,
# fontweight='bold')
# # 绘制原始数据
# ax = plt.subplot(1, 2, 1)
# plt.scatter(data[:, 0], data[:, 1], c='c')
# plt.title("data")
# # 绘制聚类后的数据和簇中心
# ax = plt.subplot(1, 2, 2)
# plt.scatter(A[:, 0], A[:, 1], c='b')
# plt.scatter(B[:, 0], B[:, 1], c='g')
# plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
# plt.title("clustered data and centroids (K = 2)")
# plt.show()


# 接下来，我们修改参数K进行聚类并得到相应的可视化。
# 例如需要将数据分为3个簇，则首先应用相同的过程对数据进行聚类，只需要修改参数(K=3)将数据分为3个簇：
ret, label, center = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 然后，当使用标签输出分离数据时，将数据分为三组：
A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]

# 最后一步是显示A、B和C，以及簇中心和训练数据：
fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14,
fontweight='bold')
# 绘制原始数据
ax = plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data")
# 绘制聚类后的数据和簇中心
ax = plt.subplot(1, 2, 2)
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')
plt.scatter(C[:, 0], C[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 3)")
plt.show()


# 我们也可以将簇数设置为4，观察算法运行结果：
# ret, label, center = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# A = data[label.ravel() == 0]
# B = data[label.ravel() == 1]
# C = data[label.ravel() == 2]
# D = data[label.ravel() == 3]

# fig = plt.figure(figsize=(12, 6))
# plt.suptitle("K-means clustering algorithm", fontsize=14,
# fontweight='bold')
# # 绘制原始数据
# ax = plt.subplot(1, 2, 1)
# plt.scatter(data[:, 0], data[:, 1], c='c')
# plt.title("data")
# # 绘制聚类后的数据和簇中心
# ax = plt.subplot(1, 2, 2)
# plt.scatter(A[:, 0], A[:, 1], c='b')
# plt.scatter(B[:, 0], B[:, 1], c='g')
# plt.scatter(C[:, 0], C[:, 1], c='r')
# plt.scatter(D[:, 0], D[:, 1], c='yellowgreen')
# plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
# plt.title("clustered data and centroids (K = 4)")
# plt.show()