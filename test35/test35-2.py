# Otsu阈值算法
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 在test35的效果图中，可以看到源图像中没有噪声，因此算法可以正常工作，
# 接下来，我们手动为图像添加噪声，以观察噪声对Otus阈值算法的影响，
# 然后利用高斯滤波消除部分噪声，以查看阈值图像变化情况：
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

# 加载图像、添加噪声，并将其转换为灰度图像
image = cv2.imread('../opencvStudy/test35/imgs/test03.jpg')
image = gasuss_noise(image,var=0.05)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# 应用 Otsu 阈值算法
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 高斯滤波
gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)
# 计算直方图
hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])
# 高斯滤波后的图像，应用 Otsu 阈值算法
ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
plt.suptitle("Otsu's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)
show_hist_with_matplotlib_gray(hist, "", 2, 'm', ret1)
show_hist_with_matplotlib_gray(hist2, "", 3, 'm', ret2)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img with noise", 4)
show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Otsu's binarization (before applying a Gaussian filter)", 5)
show_img_with_matplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR), "Otsu's binarization (after applying a Gaussian filter)", 6)

plt.show()

# 如结果图所示，如果不应用平滑滤波，阈值图像中充满了噪声，
# 应用高斯滤波后可以正确过滤掉噪声，同时可以看到滤波后得到的图像是双峰的。
# 注：本人图片导致效果有些差，可以看教程图像。