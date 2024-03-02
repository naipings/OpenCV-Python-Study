# OpenCV DNN人脸检测器
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 接下来，将多个图像馈送到网络进行前向计算输出人脸检测结果，以更好的理解 cv2.dnn.blobFromImages() 函数。
# 首先查看当cv2.dnn.blobFromImages() 函数中 crop=True 时的检测效果：
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test69/FP16/deploy.prototxt", "../opencvStudy/test69/FP16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# 加载图片，构建 blob
img_1 = cv2.imread('../opencvStudy/test69/imgs/test17.jpg')
img_2 = cv2.imread('../opencvStudy/test69/imgs/test2.jpg')
images = [img_1.copy(), img_2.copy()]
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
# 前向计算
net.setInput(blob_images)
detections = net.forward()

for i in range(0, detections.shape[2]):
    # 首先，获得检测结果所属的图像
    img_id = int(detections[0, 0, i, 0])
    # 获取预测的置信度
    confidence = detections[0, 0, i, 2]
    # 过滤置信度较低的预测
    if confidence > 0.25:
        # 获取当前图像尺寸
        (h, w) = images[img_id].shape[:2]
        # 获取检测的 (x, y) 坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # 绘制边界框和概率
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(images[img_id], (startX, startY), (endX, endY), (255, 0, 0), 9)
        cv2.putText(images[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=10)
    plt.axis('off')

show_img_with_matplotlib(img_1, "input img 1", 1)
show_img_with_matplotlib(img_2, "input img 2", 2)
show_img_with_matplotlib(images[0], "output img 1", 3)
show_img_with_matplotlib(images[1], "output img 2", 4)

plt.show()

# 接下来使用保持纵横比进行裁剪，观察裁剪后的检测结果，
# 可以看到纵横比保持的情况下，检测到的置信度更高：
# 只需修改 cv2.dnn.blobFromImages() 中的 crop 参数为 True
# blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)
