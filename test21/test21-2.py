# 不同色彩空间在皮肤分割中的不同效果

# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120069317

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Name and path of the images to load:（要加载的图像的名称和路径：）
image_names = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
path = '../opencvStudy/test21/imgs'


# Load all test images building the relative path using 'os.path.join'
# 使用“os.path.join”加载构建相对路径的所有测试图像
def load_all_test_images():
    """Loads all the test images and returns the created array containing the loaded images"""
    # 加载所有测试图像，并返回创建的包含加载图像的数组

    skin_images = []
    for index_image, name_image in enumerate(image_names):
        # Build the relative path where the current image is:（构建当前图像所在的相对路径：）
        image_path = os.path.join(path, name_image)
        # print("image_path: '{}'".format(image_path))
        # Read the image and add it (append) to the structure 'skin_images'
        # 读取图像并将其添加（附加）到结构“skin_images”中
        img = cv2.imread(image_path)
        skin_images.append(img)
    # Return all the loaded test images:（返回所有加载的测试图像：）
    return skin_images


# 可视化
def show_images(array_img, title, pos):
    for index_image, image in enumerate(array_img):
        show_with_matplotlib(image, title + "_" + str(index_image + 1), pos + index_image)


# 
def show_with_matplotlib(color_img, title, pos):
    # 将 BGR 图像转化为 RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(5, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 上下界数组
lower_hsv = np.array([0, 48, 80], dtype='uint8')
upper_hsv = np.array([20, 255, 255], dtype='uint8')

# 基于 HSV 颜色空间的皮肤检测
def skin_detector_hsv(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # 在 HSV 色彩空间中查找具有肤色的区域
    skin_region = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return skin_region

lower_hsv_2 = np.array([0, 50, 0], dtype="uint8")
upper_hsv_2 = np.array([120, 150, 255], dtype="uint8")


# 基于 HSV 颜色空间的皮肤检测
def skin_detector_hsv_2(bgr_image):
    # 将图片从 BRG 色彩空间转换到 HSV 空间
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # 在 HSV 色彩空间中查找具有肤色的区域
    skin_region = cv2.inRange(hsv_image, lower_hsv_2, upper_hsv_2)
    return skin_region

lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
upper_ycrcb = np.array([255, 173, 127], dtype="uint8")

# 基于 YCrCb 颜色空间的皮肤检测
def skin_detector_ycrcb(bgr_image):
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
    return skin_region

# 基于 bgr 颜色空间的皮肤检测的阈值设定
def bgr_skin(b, g, r):
    # 值基于论文《RGB-H-CbCr Skin Colour Model for Human Face Detection》
    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2

# 基于 bgr 颜色空间的皮肤检测
def skin_detector_bgr(bgr_image):
    h = bgr_image.shape[0]
    w = bgr_image.shape[1]

    res = np.zeros((h, w, 1), dtype='uint8')

    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 255
    
    return res

skin_detectors = {
    'ycrcb': skin_detector_ycrcb,
    'hsv': skin_detector_hsv,
    'hsv_2': skin_detector_hsv_2,
    'bgr': skin_detector_bgr
}

def apply_skin_detector(array_img, skin_detector):
    skin_detector_result = []
    for index_image, image in enumerate(array_img):
        detected_skin = skin_detectors[skin_detector](image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        skin_detector_result.append(bgr)
    return skin_detector_result

plt.figure(figsize=(15, 8))
plt.suptitle("Skin segmentation using different color spaces", fontsize=14, fontweight='bold')

# 加载图像
test_images = load_all_test_images()

# 绘制原始图像
show_images(test_images, "test img", 1)

# 对于每个图像应用皮肤检测函数
for i, key in enumerate(skin_detectors.keys()):
    show_images(apply_skin_detector(test_images, key), key, 7 + i * 6)

plt.show()


# 补充：构建skin_detectors字典将所有皮肤分割算法应用于测试图像，上例中定义了四个皮肤检测器。
# 可以使用以下方法调用皮肤分割检测函数（例如 skin_detector_ycrcb）：
# detected_skin = skin_detectors['ycrcb'](image)

# 注：可以使用多个测试图像来查看应用不同皮肤分割算法的效果，以了解这些算法在不同条件下的工作方式。