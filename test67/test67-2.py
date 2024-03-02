# 深度学习框架keras介绍与使用
# ————使用 keras 进行手写数字识别
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import keras
import keras
from keras.layers import *
from keras.models import *
from keras.datasets.mnist import load_data

# 接下来，我们使用 Keras 识别手写数字，与前面一样，首先需要构建模型：
def create_model():
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=(784,)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    # 使用分类交叉熵(categorical_crossentropy)作为损失函数和随机梯度下降作为优化器编译模型
    model.compile(optimizer=keras.optimizers.SGD(0.001), loss='categorical_crossentropy', metrics=['acc'])

    return model
# 使用 categorical_crossentropy 损失函数编译模型，该损失函数非常适合比较两个概率分布，使用随机下降（SGD）作为优化器。

# 接下来，加载 MNIST 数据集：
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

# 此外，由于我们使用的是全连接层，因此必须对加载的数据进行整形以输入网络：
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# 创建模型完成后，就可以训练模型，并保存创建的模型了，也可以评估模型在测试数据集上的表现：
# 模型创建
model = create_model()
# 模型训练
model.fit(train_x, train_y, batch_size=32, epochs=15, verbose=1)
# 模型保存
model.save("../opencvStudy/test67/mnist-model.h5")
# 评估模型在测试数据集上的表现
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
# 打印准确率
print("Accuracy: ", accuracy[1])


# 补充步骤：如果只是想加载现有模型： 
# 加载模型
# model = create_model()
# model.load_weights('../opencvStudy/test67/mnist-model.h5')

# 接下来，可以使用训练完成的模型来预测图像中的手写数字：
def load_digit(image_name):
    gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (28, 28))
    # cv2.imshow("", gray)
    # cv2.waitKey(0)
    gray = gray.reshape((1, 784))
    return gray
# 加载图片
test_digit_0 = load_digit("../opencvStudy/test67/67-imgs/digit_0.png")
test_digit_1 = load_digit("../opencvStudy/test67/67-imgs/digit_1.png")
test_digit_2 = load_digit("../opencvStudy/test67/67-imgs/digit_2.png")
test_digit_3 = load_digit("../opencvStudy/test67/67-imgs/digit_3.png")
# imgs = np.array([test_digit_0, test_digit_1, test_digit_2, test_digit_3])
# imgs = imgs.reshape(4, 784)
# 加载图片2
# test_digit_0 = load_digit("../opencvStudy/test67/67-imgs/02.jpg")
# test_digit_1 = load_digit("../opencvStudy/test67/67-imgs/03.jpg")
# test_digit_2 = load_digit("../opencvStudy/test67/67-imgs/04.jpg")
# imgs = np.array([test_digit_0, test_digit_1, test_digit_2])
# imgs = imgs.reshape(3, 784)

test_digit_4 = load_digit("../opencvStudy/test67/67-imgs/02.jpg")
test_digit_5 = load_digit("../opencvStudy/test67/67-imgs/03.jpg")
test_digit_6 = load_digit("../opencvStudy/test67/67-imgs/04.jpg")
imgs = np.array([test_digit_0, test_digit_1, test_digit_2, test_digit_3, test_digit_4, test_digit_5, test_digit_6])
imgs = imgs.reshape(7, 784)

# 预测加载图像 (由于TensorFlow从2.6版本开始删除了 predict_classes() 方法，所以应做相应修改)
# prediction_class = model.predict_classes(imgs)
prediction_class = np.argmax(model.predict(imgs), axis=1)
# 打印预测结果
print("Class: ", prediction_class)
