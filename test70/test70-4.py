# 使用 SqueezeNet 进行图像分类
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# 接下来使用 SqueezeNet 神经网络架构执行图像分类，参数量相比 AlexNet 网络架构减少了50倍，
# 对前面程序进行修改，将其加载的模型修改为 Caffe 预训练的 SqueezeNet 模型。
# (类别名、SqueezeNet 模型架构和模型权重参数均可在Github进行下载)————https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1
# net = cv2.dnn.readNetFromCaffe('squeezenet_v1.1_deploy.prototxt', "squeezenet_v1.1.caffemodel")
# # ...
# # 使用 SqueezeNet 时需要对预测结果 preds 进行整形
# preds = preds.reshape((1, len(classes)))
# indexes = np.argsort(preds[0])[::-1][:10]

# 完整代码：

# 1. 加载类的名称
rows = open('../opencvStudy/test70/ilsvrc12_Markup_files_for_datasets/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
# 2. 加载的 Caffe 模型
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test70/SqueezeNet_v1.1/deploy.prototxt", "../opencvStudy/test70/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel")
# 3. 加载输入图像，并对输入图像进行预处理获取 blob
image = cv2.imread('../opencvStudy/test70/70-imgs/panda.png')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda2.jpeg')
# image = cv2.imread('../opencvStudy/test70/70-imgs/panda3.jpg')
blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
print(blob.shape)
# 4. 将输入的 `blob` 馈送到网络，进行推理，并得到输出
net.setInput(blob)
preds = net.forward()
# 使用 SqueezeNet 时需要对预测结果 preds 进行整形
preds = preds.reshape((1, len(classes)))
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

show_img_with_matplotlib(image, "SqueezeNet and caffe pre-trained models", 1)

plt.show()

# 打印信息：
# Inference time: 12.51 ms
# 1. label: giant panda, probability: 0.9999127388
# 2. label: French bulldog, probability: 1.454020185e-05
# 3. label: dalmatian, probability: 1.112189693e-05
# 4. label: Arctic fox, probability: 8.336702194e-06
# 5. label: papillon, probability: 6.122082596e-06
# 6. label: Siamese cat, probability: 5.98946599e-06
# 7. label: Great Pyrenees, probability: 3.743496109e-06
# 8. label: toy terrier, probability: 3.365583325e-06
# 9. label: kuvasz, probability: 3.122086355e-06
# 10. label: Samoyed, probability: 2.496433353e-06

# 翻译即为：
# 推理时间：12.51ms
# 1.标签：大熊猫，概率：0.9999127388
# 2.标签：法国斗牛犬，概率：1.454020185e-05
# 3.标签：达尔马提亚，概率：1.112189693e-05
# 4.标签：北极狐，概率：8.336702194e-06
# 5.标签：乳头状瘤，概率：6.122082596e-06
# 6.标签：暹罗猫，概率：5.98946599e-06
# 7.标签：大比利牛斯山，概率：3.743496109e-06
# 8.标签：玩具梗，概率：3.365583325e-06
# 9.标签：kuvasz，概率：3.122086355e-06
# 10.标签：萨摩耶，概率：2.496433353e-06

# 根据打印信息可知，我们可以测试以上类别的事物。（而且只打印了最高的概率从高到低的10个类别）