# 使用dlib进行人脸识别
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 计算 128D 描述符用于量化人脸：
# 使用 dlib 库加载特征点预测器、人脸编码和人脸检测器
pose_predictor_5_point = dlib.shape_predictor("../opencvStudy/test65/Facial_feature_point_detection_detector/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("../opencvStudy/test65/dlib-models/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """返回图像中每个人脸的 128D 描述符"""

    # 检测人脸
    face_locations = detector(face_image, number_of_times_to_upsample)
    # 检测面部特征点
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # 使用每个检测到的特征点计算每个检测到的人脸的编码
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]

# 加载图像并转换为 RGB 模式
image = cv2.imread("../opencvStudy/test65/imgs/test2.jpg")
rgb = np.ascontiguousarray(image[:, :, ::-1])

# 计算图像中每个人脸的编码
encodings = face_encodings(rgb)
# 打印第一个编码的特征
print(encodings[0])


# 接下来，我们使用5张已知图片与另一张测试图像进行比较。
# 为了比较人脸，我们需要编写两个函数：compare_faces() 和 compare_faces_ordered()。
# compare_faces() 函数返回已知人脸编码与待识别人脸间的距离：
def compare_faces(encodings, encoding_to_check):
    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))

# compare_faces_ordered() 函数返回排序后的已知人脸编码与待识别人脸间的距离和相应的名称：
def compare_faces_ordered(encodings, face_names, encoding_to_check):
    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))

# 接下来，将5个已标记图像与1个未标记图像进行比较，
# 第一步是加载所有图像并转换为 RGB 格式：
# 加载图像
known_image_1 = cv2.imread("../opencvStudy/test65/65-imgs/02_1.jpg")
known_image_2 = cv2.imread("../opencvStudy/test65/65-imgs/01_1.jpg")
known_image_3 = cv2.imread("../opencvStudy/test65/65-imgs/01_2.jpg")
known_image_4 = cv2.imread("../opencvStudy/test65/65-imgs/01_3.jpg")
known_image_5 = cv2.imread("../opencvStudy/test65/65-imgs/02_2.jpg")
unknown_image = cv2.imread("../opencvStudy/test65/65-imgs/test.jpg")

# 灰度图像用于 HOG 人脸检测
gray_image_1 = cv2.cvtColor(known_image_1, cv2.COLOR_BGR2GRAY)
gray_image_2 = cv2.cvtColor(known_image_2, cv2.COLOR_BGR2GRAY)
gray_image_3 = cv2.cvtColor(known_image_3, cv2.COLOR_BGR2GRAY)
gray_image_4 = cv2.cvtColor(known_image_4, cv2.COLOR_BGR2GRAY)
gray_image_5 = cv2.cvtColor(known_image_5, cv2.COLOR_BGR2GRAY)
unknown_gray_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
# 执行检测
rects_1 = detector(gray_image_1, 0)
rects_2 = detector(gray_image_2, 0)
rects_3 = detector(gray_image_3, 0)
rects_4 = detector(gray_image_4, 0)
rects_5 = detector(gray_image_5, 0)
unknown_rects = detector(unknown_gray_image, 0)
# 可视化检测结果
def show_detection(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 5)
    return image
# 绘制检测框
img_faces_1 = show_detection(known_image_1.copy(), rects_1)
img_faces_2 = show_detection(known_image_2.copy(), rects_2)
img_faces_3 = show_detection(known_image_3.copy(), rects_3)
img_faces_4 = show_detection(known_image_4.copy(), rects_4)
img_faces_5 = show_detection(known_image_5.copy(), rects_5)
unknown_img_faces = show_detection(unknown_image.copy(), unknown_rects)

# 转换为 RGB 格式
known_image_1 = np.ascontiguousarray(known_image_1[:, :, ::-1])
known_image_2 = np.ascontiguousarray(known_image_2[:, :, ::-1])
known_image_3 = np.ascontiguousarray(known_image_3[:, :, ::-1])
known_image_4 = np.ascontiguousarray(known_image_4[:, :, ::-1])
known_image_5 = np.ascontiguousarray(known_image_5[:, :, ::-1])
unknown_image = np.ascontiguousarray(unknown_image[:, :, ::-1])
# 标记人脸
names = ["02_1.jpg", "01_1.jpg", "01_2.jpg", "01_3.jpg", "02_2.jpg"]


# 下一步是计算每个图像的 128D 编码：
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_image_5_encoding = face_encodings(known_image_5)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding, known_image_5_encoding]
unknown_encoding = face_encodings(unknown_image)[0]

# 最后，可以使用compare_faces_ordered() 函数与待识别人脸比较：
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)
# 打印返回信息
print()
# print(computed_distances)
print(computed_distances_ordered)
print(ordered_names)

# 根据打印结果可知：
# 因为02_1.jpg 与 01_2.jpg 对应的 128D 描述符更小，所以test.jpg与它们更相似，所以（即：）test.jpg为02_x.jpg系列的图片。
'''打印结果：
(0.31224942907393355, 0.3655018426948874, 0.4899680125325854, 0.498395703590526, 0.8370318567506712)
('02_1.jpg', '02_2.jpg', '01_1.jpg', '01_3.jpg', '01_2.jpg')
'''

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face recognition using dlib", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces_1, "img_faces_1(gray, 0)", 1)
show_img_with_matplotlib(img_faces_2, "img_faces_2(gray, 0)", 2)
show_img_with_matplotlib(img_faces_3, "img_faces_3(gray, 0)", 3)
show_img_with_matplotlib(img_faces_4, "img_faces_4(gray, 0)", 4)
show_img_with_matplotlib(img_faces_5, "img_faces_5(gray, 0)", 5)
show_img_with_matplotlib(unknown_img_faces, "unknown_img_faces(gray, 0)", 6)

plt.show()