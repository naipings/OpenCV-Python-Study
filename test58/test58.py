# 使用cvlib进行人脸检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import cvlib as cv

# 加载图像
img = cv2.imread('../opencvStudy/test58/imgs/test17.jpg')

# cvlib中提供了 detect_face() 函数用于检测人脸：
# 第四种方法同样需要且仅需要一行代码
faces, confidences = cv.detect_face(img)
# 绘制边界框
def show_detection(image, faces):
    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    return image
# # 显示检测结果
img_result = show_detection(img.copy(), faces)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using cvlib face detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_result, "cvlib face detector: " + str(len(faces)), 1)

plt.show()
