# 使用OpenCV进行人脸识别
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# OpenCV提供了三种不同的实现来执行人脸识别：
# 我们仅需要更改识别器的创建方式就可以独立于其内部算法使用它们：
# 创建识别器
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()
# 一旦创建，其将独立于特定内部算法，均可以分别使用方法 train() 和 predict() 来执行人脸识别系统的训练和测试，使用这些方法的方式与创建的识别器无关。
# 因此，可以很容易对比这三种识别器，然后针对特定任务选择性能最佳的识别器。
# 但在涉及不同环境和光照条件的户外识别图像时，LBPH 通常比另外两种方法具有更好的性能。
# 此外，LBPH 人脸识别器支持 update() 函数，可以在该方法中根据新数据更新人脸识别器，但 Eigenfaces 和 Fisherfaces 方法并不支持 update() 函数。

# 为了训练识别器，应该调用train() 方法：
face_recognizer.train(faces, labels)
# cv2.face_FaceRecognizer.train(src, labels) 方法训练具体的人脸识别器，其中 src 表示图像（人脸）训练集，参数 labels 为训练集中的每张图像对应的标签。

# 要识别新面孔，应调用predict() 方法：
label, confidence = face_recognizer.predict(face)
# cv2.face_FaceRecognizer.predict() 方法通过输出预测的标签和相应的置信度来输出（预测）对新 src 图像的识别结果。

# OpenCV还提供 write() 和 read() 方法用于保存创建的模型和加载之前创建的模型。
# 对于这两种方法，文件名参数设置要保存或加载的模型的名称：
cv2.face_FaceRecognizer.write(filename)
cv2.face_FaceRecognizer.read(filename)

# 如果使用的是 LBPH 人脸识别器，可以使用 update() 方法进行更新：
cv2.face_FaceRecognizer.update(src, labels)
# 其中，src 和 labels 设置了用于更新 LBPH 识别器的新训练数据集。