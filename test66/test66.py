# 使用face_recognition进行人脸识别
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib
import face_recognition

# 加载图像
known_image_1 = face_recognition.load_image_file("../opencvStudy/test66/65-imgs/02_1.jpg")
known_image_2 = face_recognition.load_image_file("../opencvStudy/test66/65-imgs/01_1.jpg")
known_image_3 = face_recognition.load_image_file("../opencvStudy/test66/65-imgs/01_3.jpg")
known_image_4 = face_recognition.load_image_file("../opencvStudy/test66/65-imgs/02_2.jpg")

# 为每个图像创建标签
names = ["02_1.jpg", "01_1.jpg", "01_3.jpg", "02_2.jpg"]

# 加载待识别图像(用于与已加载的标记图像进行比较)
unknown_image = face_recognition.load_image_file("../opencvStudy/test66/65-imgs/test.jpg")

# 人脸检测部分：
detector = dlib.get_frontal_face_detector()
# 灰度图像用于 HOG 人脸检测
gray_image_1 = cv2.cvtColor(known_image_1, cv2.COLOR_BGR2GRAY)
gray_image_2 = cv2.cvtColor(known_image_2, cv2.COLOR_BGR2GRAY)
gray_image_3 = cv2.cvtColor(known_image_3, cv2.COLOR_BGR2GRAY)
gray_image_4 = cv2.cvtColor(known_image_4, cv2.COLOR_BGR2GRAY)
unknown_gray_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
# 执行检测
rects_1 = detector(gray_image_1, 0)
rects_2 = detector(gray_image_2, 0)
rects_3 = detector(gray_image_3, 0)
rects_4 = detector(gray_image_4, 0)
unknown_rects = detector(unknown_gray_image, 0)
# 可视化检测结果
def show_detection(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 5)
    return image
# 绘制检测框
img_faces_1 = show_detection(known_image_1.copy(), rects_1)
img_faces_2 = show_detection(known_image_2.copy(), rects_2)
img_faces_3 = show_detection(known_image_3.copy(), rects_3)
img_faces_4 = show_detection(known_image_4.copy(), rects_4)
unknown_img_faces = show_detection(unknown_image.copy(), unknown_rects)

# 将每张图片中的人脸编码为 128D 向量
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding,
known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# 人脸对比
results = face_recognition.compare_faces(known_encodings, unknown_encoding)
# 打印结果
print(results)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    # img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 4, pos)
    plt.imshow(color_img)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face recognition using dlib", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces_1, "img_faces_1(gray, 0)", 1)
show_img_with_matplotlib(img_faces_2, "img_faces_2(gray, 0)", 2)
show_img_with_matplotlib(img_faces_3, "img_faces_3(gray, 0)", 3)
show_img_with_matplotlib(img_faces_4, "img_faces_4(gray, 0)", 4)
show_img_with_matplotlib(unknown_img_faces, "unknown_img_faces(gray, 0)", 6)

plt.show()
