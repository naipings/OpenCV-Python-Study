# 支持向量机示例
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 图像可视化用：
def show_with_matplotlib(img, title):
    # 将 BGR 图像转换为 RGB
    img_RGB = img[:, :, ::-1]
    # 使用 Matplotlib 显示图形
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()

# 首先，需要创建训练数据和标签
labels = np.array([1, 1, -1, -1, -1])
data = np.matrix([[800, 40], [850, 400], [500, 10], [550, 300], [450, 600]], dtype=np.float32)
# 以上代码创建了5个点，前两个点被指定为1类，而另外3个被指定为-1类。

# 接下来使用svm_init()函数初始化SVM模型：
def svm_init(C=12.5, gamma=0.50625):
    """ 创建 SVM 模型并为其分配主要参数，返回模型 """
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return model
# 初始化 SVM 模型
svm_model = svm_init(C=12.5, gamma=0.50625)
# 创建的SVM核类型设置为LINEAR，SVM的类型设置为C_SVC。

# 然后，编写svm_train()函数训练SVM模型：
def svm_train(model, samples, responses):
    # 使用 samples 和 responses 训练模型
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model
# 训练 SVM
svm_train(svm_model, data, labels)

# 然后创建一个图像，并绘制SVM响应：
def svm_predict(model, samples):
    """根据训练好的模型预测响应"""

    return model.predict(samples)[1].ravel()

def show_svm_response(model, image):

    colors = {1: (255, 255, 0), -1: (0, 255, 255)}

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sample = np.matrix([[j, i]], dtype=np.float32)
            response = svm_predict(model, sample)

            image[i, j] = colors[response.item(0)]
    
    cv2.circle(image, (800, 40), 10, (255, 0, 0), -1)
    cv2.circle(image, (850, 400), 10, (255, 0, 0), -1)

    cv2.circle(image, (500, 10), 10, (0, 255, 0), -1)
    cv2.circle(image, (550, 300), 10, (0, 255, 0), -1)
    cv2.circle(image, (450, 600), 10, (0, 255, 0), -1)

    support_vectors = model.getUncompressedSupportVectors()
    for i in range(support_vectors.shape[0]):
        cv2.circle(image, (int(support_vectors[i, 0]), int(support_vectors[i, 1])), 15, (0, 0, 255), 6)
    
    # 调用show_with_matplotlib()函数显示图像：
    plt.suptitle("SVM introduction", fontsize=14, fontweight='bold')
    show_with_matplotlib(image, 'Visual representation of SVM model')
# 创建图像
img_output = np.zeros((640, 1200, 3), dtype="uint8")
# 显示 SVM 响应
show_svm_response(svm_model, img_output)

# 根据运行结果可知，SVM使用训练数据进行了训练，可用于对图像中所有点进行分类。
# SVM将图像划分为黄色和青色区域，可以看到两个区域的边界对应于两个类之间的最佳间隔，因为到两个类中最近元素的距离最大，支持向量用红线边框显示。
