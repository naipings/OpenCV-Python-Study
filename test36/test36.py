# Triangle阈值算法
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 还有另一种自动阈值算法称为Triangle阈值算法，
# 它是一种基于形状的方法，因为它分析直方图的结构或形状(例如，谷、峰和其他直方图形状特征)。
# 该算法可以分为三步：
# 第一步，计算直方图灰度轴上最大值bmax 和 灰度轴上最小值bmin 之间的直线；
# 第二步，计算 b[ bmin , bmax ] 范围内直线(在第一步中计算)到直方图的距离；
# 最后，选择直方图与直线距离最大的水平作为阈值。
# OpenCV中 Triangle阈值算法的使用方式与 Otsu的算法非常相似，只需要将 cv2.THRESH_OTSU标志修改为 cv2.THRESH_TRIANGLE：

# 手动为图像添加噪声,以观察噪声对Triangle阈值算法的影响，（可见test35-2.py）
def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

# 显示灰度直方图
def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    # 可视化阈值
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

# 加载图像
image = cv2.imread('../opencvStudy/test36/imgs/test03.jpg')
image = gasuss_noise(image,var=0.05)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
# 高斯滤波
gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)

hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])
ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Triangle's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)
show_hist_with_matplotlib_gray(hist, "", 2, 'm', ret1)
show_hist_with_matplotlib_gray(hist2, "", 3, 'm', ret2)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img with noise", 4)
show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Triangle's binarization (before applying a Gaussian filter)", 5)
show_img_with_matplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR), "Triangle's binarization (after applying a Gaussian filter)", 6)

plt.show()

# 在效果图中，我们可以看到将Triangle阈值算法应用于噪声图像时的输出。