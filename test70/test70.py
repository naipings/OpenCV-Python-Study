# OpenCV图像分类
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# 接下来将介绍使用不同的预训练深度学习模型执行图像分类，
# 为了对比不同模型的运行效率，可以使用 net.getPerfProfile() 方法获取推理阶段所用时间：
# 前向计算获取预测结果
net.setInput(blob)
preds = net.forward()
# 获取推理时间
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
# 如上所示，在执行推理过后调用 net.getPerfProfile() 方法获取推理时间，通过这种方式，可以比较使用不同深度学习架构的推理时间。
