# 基于HOG特征和滑动窗口的人脸检测器
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 第一步是从dlib上加载人脸检测器：
# 加载人脸检测器
# 第二种方法的第一行代码
detector = dlib.get_frontal_face_detector()
# 加载图像并转换为灰度图像
img = cv2.imread('../opencvStudy/test56/imgs/test17.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 下一步是执行检测：
# 第二种方法的第二行代码，至此第二种方法也讲解完毕了
rects_1 = detector(gray, 0)
# rects_2 = detector(gray, 1)
rects_2 = detector(gray, 3)
# detector() 的第二个参数表示在执行检测过程之前对图像进行上采样的次数，
# 因为图像越大检测器检测到更多的人脸的可能性就越高，但执行时间相应也会增加。

# 最后可视化检测结果：
def show_detection(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 5)
    return image
# 绘制检测框
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# 绘制图像
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using dlib DNN face detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces_1, "detector(gray, 0): " + str(len(rects_1)), 1)
# show_img_with_matplotlib(img_faces_2, "detector(gray, 1): " + str(len(rects_2)), 2)
show_img_with_matplotlib(img_faces_2, "detector(gray, 3): " + str(len(rects_2)), 2)

plt.show()

# 小样本训练可以参考（结合教程）