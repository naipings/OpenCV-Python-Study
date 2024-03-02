# 使用 GoogLeNet 进行图像分类
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# 使用 GoogLeNet 模型进行图像分类的步骤与使用 Caffe 预训练的 AlexNet 模型进行图像分类的步骤相同，
# 唯一的区别在于其加载的模型为 Caffe 预训练的 GoogLeNet 模型。
# (类别名、GoogLeNet 模型架构和模型权重参数均可在Github进行下载)————https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# 完整代码：

# 1. 加载类的名称
rows = open('../opencvStudy/test70/ilsvrc12_Markup_files_for_datasets/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
# 2. 加载的 Caffe 模型
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test70/googleNet/deploy.prototxt", "../opencvStudy/test70/googleNet/bvlc_googlenet.caffemodel")
# 3. 加载输入图像，并对输入图像进行预处理获取 blob
image = cv2.imread('../opencvStudy/test70/70-imgs/panda.png')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda2.jpeg')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda3.jpg')
blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
print(blob.shape)
# 4. 将输入的 `blob` 馈送到网络，进行推理，并得到输出
net.setInput(blob)
preds = net.forward()
# 获取推理时间
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
# 5. 得到概率最高的 10 个预测类别(降序排列)
indexes = np.argsort(preds[0])[::-1][:10]
# 6. 在图像上绘制置信度最高的类别和概率
text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]], preds[0][indexes[0]] * 100)
y0, dy = 30, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# 打印置信度排名前十的类别
for (index, idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.10}".format(index + 1, classes[idx], preds[0][idx]))

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=10)
    plt.axis('off')

show_img_with_matplotlib(image, "GoogLeNet and caffe pre-trained models", 1)

plt.show()

# 打印信息：
# Inference time: 35.51 ms
# 1. label: giant panda, probability: 0.999854207
# 2. label: soccer ball, probability: 0.0001214441727
# 3. label: ice bear, probability: 8.52100402e-06
# 4. label: lesser panda, probability: 4.296696716e-06
# 5. label: dalmatian, probability: 3.101016773e-06
# 6. label: French bulldog, probability: 1.789065777e-06
# 7. label: hog, probability: 1.084415544e-06
# 8. label: brown bear, probability: 1.008830623e-06
# 9. label: Walker hound, probability: 8.485408216e-07
# 10. label: paper towel, probability: 4.166564906e-07

# 翻译即为：
# 推理时间：35.51毫秒
# 1.标签：大熊猫，概率：0.999854207
# 2.标签：足球，概率：0.0001214441727
# 3.标签：冰熊，概率：8.52100402e-06
# 4.标签：小熊猫，概率：4.296696716e-06
# 5.标签：达尔马提亚，概率：3.101016773e-06
# 6.标签：法国斗牛犬，概率：1.789065777e-06
# 7.标签：hog，概率：1.084415544e-06
# 8.标签：棕熊，概率：1.008830623e-06
# 9.标签：Walker hound，概率：8.485408216e-07
# 10.标签：纸巾，概率：4.166564906e-07

# 根据打印信息可知，我们可以测试以上类别的事物。（而且只打印了最高的概率从高到低的10个类别）