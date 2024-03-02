# 使用face_recognition进行人脸检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import face_recognition

# 加载图像
img = cv2.imread('../opencvStudy/test57/imgs/test17.jpg')

# 使用face_recognition检测人脸，仅需要调用face_locations() 函数：
# 第三种方法需要且仅需要一行代码
rects_1 = face_recognition.face_locations(img, 0, "hog")
rects_2 = face_recognition.face_locations(img, 1, "hog")
# face_locations() 的第一个参数是输入图像（RGB）；第二个参数设置在执行检测之前对输入图像上采样的次数；第三个参数是要使用的人脸检测模型。

# 上述示例使用hog检测模型，此外face_locations() 也可以配置为使用cnn人脸检测器检测人脸：
# 使用 cnn 人脸检测器检测人脸
rects_3 = face_recognition.face_locations(img, 0, "cnn")
rects_4 = face_recognition.face_locations(img, 1, "cnn")

# 最后可视化检测结果：
def show_detection(image, faces):
    for face in faces:
        top, right, bottom, left = face
        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 10)
    return image
# 显示检测结果
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)
img_faces_3 = show_detection(img.copy(), rects_3)
img_faces_4 = show_detection(img.copy(), rects_4)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using face_recognition HOG and CNN face detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces_1, "face_locations(rgb, 0, hog): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "face_locations(rgb, 1, hog): " + str(len(rects_2)), 2)
show_img_with_matplotlib(img_faces_3, "face_locations(rgb, 0, cnn): " + str(len(rects_3)), 3)
show_img_with_matplotlib(img_faces_4, "face_locations(rgb, 1, cnn): " + str(len(rects_4)), 4)

plt.show()

