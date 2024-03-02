# 使用face_recognition检测面部特征点
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import face_recognition

# 如果想要使用face_recognition包检测和绘制面部特征点，需要调用 face_recognition.face_landmarks() 函数：
# 加载图像
image = cv2.imread("../opencvStudy/test60/imgs/test2.jpg")
image_68 = image.copy()
image_5 = image.copy()
# 将图像从 BGR 颜色转换为 RGB 颜色
rgb = image[:, :, ::-1]
# 检测 68 个特征点
# 第三种面部特征点检测方法，第一行代码
face_landmarks_list_68 = face_recognition.face_landmarks(rgb)
# 此函数返回图像中每张脸的面部特征点（例如，眼睛和鼻子）的字典，如果打印检测到的特征点，则可以看到输出如下：
print(face_landmarks_list_68)
'''输出如下：
[{'chin': [(790, 1060), (784, 1132), (790, 1200), (801, 1263), (818, 1324), (850, 1379), (887, 1428), (931, 1470), (982, 1489), (1038, 1485), (1094, 1453), (1145, 1416), (1192, 1374), (1228, 1324), (1260, 1272), (1290, 1214), (1310, 1149)], 'left_eyebrow': [(843, 1014), (893, 1004), (940, 1008), (985, 1021), (1024, 1048)], 'right_eyebrow': [(1119, 1068), (1166, 1058), (1213, 1061), (1258, 1072), (1292, 1099)], 'nose_bridge': [(1063, 1126), (1055, 1180), (1047, 1235), (1039, 1288)], 'nose_tip': [(996, 1298), (1012, 1310), (1030, 1323), (1052, 1319), (1073, 1313)], 'left_eye': [(885, 1101), (919, 1083), (961, 1091), (991, 1128), (952, 1132), (910, 1125)], 'right_eye': [(1127, 1151), (1163, 1125), (1205, 1128), (1235, 1152), (1205, 1170), (1162, 1166)], 'top_lip': [(937, 1347), (972, 1350), (1001, 1350), (1023, 1361), (1045, 1357), (1069, 1367), (1097, 1377), (1080, 1380), (1042, 1379), (1020, 1378), (997, 1371), (953, 1355)], 'bottom_lip': [(1097, 1377), (1061, 1404), (1034, 1411), (1011, 1410), (987, 1401), (962, 1381), (937, 1347), (953, 1355), (995, 1364), (1017, 1372), (1039, 1372), (1080, 1380)]}]
'''

# 最后绘制检测到的特征点：
for face_landmarks in face_landmarks_list_68:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_68, p, 4, (255, 0, 0), 4)


# face_recognition.face_landmarks() 函数的用法如下：
# face_landmarks(face_image, face_locations=None, model="large")
# 默认情况下会检测到68个特征点，如果model="small"，只会检测 5 个特征点：
# 检测 5 个特征点
face_landmarks_list_5 = face_recognition.face_landmarks(rgb, None, "small")

# 如果打印返回的结果 face_landmarks_list_5，可以得到以下输出：
print()
print(face_landmarks_list_5)
'''输出如下：
[{'nose_tip': [(1027, 1321)], 'left_eye': [(881, 1095), (991, 1127)], 'right_eye': [(1235, 1152), (1126, 1149)]}]
'''
# 在这种情况下，结果字典只包含双眼和鼻尖的面部特征点位置。

# 接下来，同样绘制检测到的特征点：
for face_landmarks in face_landmarks_list_5:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_5, p, 4, (255, 0, 0), 4)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Facial landmarks datection using face_recognition", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image_68, "68 facial landmarks", 1)
show_img_with_matplotlib(image_5, "5 facial landmarks", 2)

plt.show()

# 运行程序可以看到，使用 face_recognition包检测到的 68 个和 5 个面部特征点。