# 简单的阈值技术
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 在test32.py中,我们已经见到OpenCV中提供的简单阈值处理函数————cv2.threshold()，
# 该函数用法如下：
# cv2.threshold(src, thresh, maxval, type, dst=None) -> retval, dst

# cv2.threshold()函数对src输入数组（可为单通道或多通道图像）应用预定义常数thresh设置的阈值；
# type参数用于设置阈值类型，阈值类型的可选值如下：
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV
# cv2.THRESH_OTSU
# cv2.THRESH_TRIANGLE
# maxval 参数用于设置最大值，其仅在阈值类型为 cv2.THRESH_BINARY 和 cv2.THRESH_BINARY_INV 时有效；
# 需要注意的是，在阈值类型为 cv2.THRESH_OTSU 和 cv2.THRESH_TRIANGLE 时，输入图像 src 应为为单通道。

# 有关每种阈值类型的具体公式以及分析，参见：https://blog.csdn.net/LOVEmy134611/article/details/120069509

# 接下来使用不同阈值类型对同样的测试图像进行阈值处理，观察不同的阈值处理效果：
# 构建测试图像（可见test32.py）
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

# 使用 build_sample_image() 函数构建测试图像
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret1, thresh1 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(gray_image, 100, 220, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
ret4, thresh4 = cv2.threshold(gray_image, 100, 220, cv2.THRESH_BINARY_INV)
ret5, thresh5 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TRUNC)
ret6, thresh6 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO)
ret7, thresh7 = cv2.threshold(gray_image,100,255, cv2.THRESH_TOZERO_INV)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(8, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Thresholding introduction", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "THRESH_BINARY - thresh = 100 & maxValue = 255", 2)
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "THRESH_BINARY - thresh = 100 & maxValue = 220", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "THRESH_BINARY_INV - thresh = 100", 4)
# 其他图像可视化方法类似，不再赘述
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "THRESH_BINARY_INV - thresh = 100 & maxValue = 220", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "THRESH_TRUNC - thresh = 100", 6)
show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "THRESH_TOZERO - thresh = 100", 7)
show_img_with_matplotlib(cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "THRESH_TOZERO_INV - thresh = 100", 8)

# 可视化测试图像
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img with tones of gray - left to right: (0,50,100,150,200,250)", 1)

plt.show()

# 如上图所示，maxval参数仅在使用cv2.THRESH_BINARY 和 cv2.THRESH_BINARY_INV阈值类型时有效，
# 上例中将cv2.THRESH_BINARY 和 cv2.THRESH_BINARY_INV类型的maxval值设置为255及220，以便查看阈值图像在这两种情况下的变化情况。