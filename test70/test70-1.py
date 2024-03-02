# 使用 AlexNet 进行图像分类
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# 使用 Caffe 预训练的 AlexNet 模型进行图像分类可以分为以下步骤：
# 1.加载类别的名称
# 2.加载 Caffe 模型
# 3.加载输入图像，并对输入图像进行预处理获取 blob
# 4.将输入的 blob 馈送到网络，进行推理，并得到输出
# 5.得到概率最高的 10 个预测类别(降序排列)
# 6.在图像上绘制置信度最高的类别和概率

# 1. 加载类的名称
rows = open('../opencvStudy/test70/ilsvrc12_Markup_files_for_datasets/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
# 2. 加载的 Caffe 模型
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test70/bvlc_alexnet/deploy.prototxt", "../opencvStudy/test70/bvlc_alexnet/bvlc_alexnet.caffemodel")
# 3. 加载输入图像，并对输入图像进行预处理获取 blob
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda.png')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda2.jpeg')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda3.jpg')
image = cv2.imread('../opencvStudy/test70/70-imgs/ship3.jpeg')
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

show_img_with_matplotlib(image, "AlexNet and caffe pre-trained models", 1)

plt.show()

# 打印信息：
# Inference time: 30.23 ms
# 1. label: giant panda, probability: 0.995234549
# 2. label: Siamese cat, probability: 0.002214594278
# 3. label: French bulldog, probability: 0.0003789721814
# 4. label: American Staffordshire terrier, probability: 0.0003668769205
# 5. label: soccer ball, probability: 0.0002231821854
# 6. label: Cardigan, probability: 0.000163433011
# 7. label: Labrador retriever, probability: 0.0001266862819
# 8. label: dalmatian, probability: 0.0001146125142
# 9. label: beagle, probability: 9.322020924e-05
# 10. label: guinea pig, probability: 6.114101416e-05

# 翻译即为：
# 推断时间：30.23毫秒
# 1.标签：大熊猫，概率：0.9952334549
# 2.标签：暹罗猫，概率：0.00214594278
# 3.标签：法国斗牛犬，概率：0.0003789721814
# 4.标签：美国斯塔福德郡梗，概率：0.0003668769205
# 5.标签：足球，概率：0.0002231821854
# 6.标签：卡迪根，概率：0.000163433011
# 7.标签：拉布拉多寻回犬，概率：0.0001266862819
# 8.标签：达尔马提亚，概率：0.0001146125142
# 9.标签：比格犬，概率：9.322020924e-05
# 10.标签：豚鼠，概率：6.114101416e-05

# 根据打印信息可知，我们可以测试以上类别的事物。（而且只打印了最高的概率从高到低的10个类别）