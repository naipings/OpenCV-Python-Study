# 基于Haar级联的人脸检测器
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 加载图片
img = cv2.imread('../opencvStudy/test55/imgs/test2.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test1.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test3.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test4.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test5.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test6.jpg')
# img = cv2.imread('../opencvStudy/test55/imgs/test01.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OpenCV提供了 cv2.CascadeClassifier() 函数用于从文件中加载分类器：
# 加载级联分类器
# 第一种方法的第一行代码
cas_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
cas_alt = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt.xml")
cas_alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt_tree.xml")
cas_face = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalcatface.xml")

# 接下来，就可以使用cv2.CascadeClassifier.detectMultiScale() 函数执行检测了：
# 第一种方法的第二行代码
# 这样就完成了第一种方法的介绍了
faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)
faces_alt = cas_alt.detectMultiScale(gray)
faces_alt_tree = cas_alt_tree.detectMultiScale(gray)
faces_face = cas_face.detectMultiScale(gray)

# cv2.CascadeClassifier.detectMultiScale() 函数检测对象并将它们作为矩形列表返回。
# 为了进行可视化，最后编写show_detection() 函数进行可视化：（也就是绘制矩形）
def show_detection(image, faces):
    """在每个检测到的人脸上绘制一个矩形进行标示"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image
# 调用 show_detection() 函数标示检测到的人脸
img_faces_alt2 = show_detection(img.copy(), faces_alt2)
img_faces_default = show_detection(img.copy(), faces_default)
img_faces_alt = show_detection(img.copy(), faces_alt)
img_faces_alt_tree = show_detection(img.copy(), faces_alt_tree)
img_faces_face = show_detection(img.copy(), faces_face)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using haar feature-based cascade classifiers", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces_alt2, "haarcascade_frontalface_alt2.xml", 1)
show_img_with_matplotlib(img_faces_default, "haarcascade_frontalface_default.xml", 2)
show_img_with_matplotlib(img_faces_alt, "haarcascade_frontalface_alt.xml", 3)
show_img_with_matplotlib(img_faces_alt_tree, "haarcascade_frontalface_alt_tree.xml", 4)
show_img_with_matplotlib(img_faces_face, "haarcascade_frontalcatface.xml", 5)

# plt.show()


# 拓展，用一行代码调用检测器检测人脸，这就是OpenCV提供的 cv2.face.getFacesHAAR() 函数：
# 一行代码实现基于 Haar 级联的人脸检测器，学一送一
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
retval, faces_haar_default = cv2.face.getFacesHAAR(img, cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

# 注：cv2.CascadeClassifier.detectMultiScale() 函数需要灰度图像作为输入，而cv2.face.getFacesHAAR()需要BGR图像作为输入。
# 此外，cv2.CascadeClassifier.detectMultiScale()将检测到的人脸输出为矩形列表，例如，如果检测到两个脸，则输出形式如下：
# [[809 494 152 152] [168 503 188 188]]

# 而，cv2.face.getFacesHAAR()函数则以以下格式返回检测到的人脸：
# [[[ 809  493  151  151]] [[ 167  503  189  189]]]

# 因此，如果使用cv2.face.getFacesHAAR()函数进行n检测，绘制检测框时要调用np.squeeze()函数消除多余维度：
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
faces_haar_alt2 = np.squeeze(faces_haar_default)

# 调用 show_detection() 函数标示检测到的人脸
img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
img_faces_haar_default = show_detection(img.copy(), faces_haar_alt2)

# 可视化
show_img_with_matplotlib(img_faces_haar_alt2, "haarcascade_frontalface_alt2.xml", 7)
show_img_with_matplotlib(img_faces_haar_default, "haarcascade_frontalface_default.xml", 8)

plt.show()

# 还需要说明，cv2.CascadeClassifier.detectMultiScale() 函数有minSize和maxSize参数，用以设置最小尺寸(小于minSize的对象不会被检测)和最大尺寸(大于maxSize的对象不被检测到)，而cv2.face.getFacesHAAR()函数并不提供此参数。
