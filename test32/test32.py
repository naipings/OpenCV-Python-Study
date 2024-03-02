# 阈值技术简介

# 详细参见：https://blog.csdn.net/LOVEmy134611/article/details/120069509
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 阈值处理是一种简单、有效的将图像划分为前景和背景的方法。图像分割通常用于根据对象的某些属性（例如，颜色、边缘或直方图）从背景中提取对象。
# 最简单的阈值方法会利用预定义常数（阈值），如果像素强度小于阈值，则用黑色像素替换；如果像素强度大于阈值，则用白色像素替换。
# OpenCV提供了cv2.threshold()函数进行阈值处理。

# 为了测试cv2.threshold()函数，首次创建测试图像，其包含一些填充了不同的灰色调的大小相同的区域，利用build_sample_image()函数构建此测试图像：
def build_sample_image():
    """创建填充了不同的灰色调的大小相同的区域，作为测试图像"""
    # 定义不同区域
    tones = np.arange(start=50, stop=300, step=50)
    # 初始化
    result = np.zeros((50, 150, 3), dtype="uint8")

    for tone in tones:
        img = np.ones((50, 150, 3), dtype="uint8") * tone
        # 沿轴连接数组
        result = np.concatenate((result, img), axis=1)
    return result

# 接下来将使用不同的预定义阈值：0、50、100、150、200和250调用cv2.threshold()函数，以查看不同预定义阈值对阈值图像影响。
# 例如，使用阈值thresh=50对图像进行阈值处理：
# ret1, thresh1 = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# 其中，thresh1仅是含黑白色的阈值图像。源图像gray_image中灰色强度小于50的像素为黑色，强度大于50的像素为白色。
# 使用多个不同阈值对图像进行阈值处理：
# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(7, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Thresholding introduction", fontsize=14, fontweight='bold')

# 使用 build_sample_image() 函数构建测试图像
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
for i in range(6):
    # 使用多个不同阈值对图像进行阈值处理
    ret, thresh = cv2.threshold(gray_image, 50 * i, 255, cv2.THRESH_BINARY)
    # 可视化
    show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = {}".format(i * 50), i + 2)
# 可视化测试图像
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img with tones of gray - left to right: (0,50,100,150,200,250)", 1)

plt.show()

# 从效果图我们可以看出，根据阈值和样本图像灰度值的不同，阈值处理后生成的黑白图像的变化情况。