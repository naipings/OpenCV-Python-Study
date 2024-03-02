# 深度学习框架keras介绍与使用
# ————使用 keras 实现线性回归模型
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

# 产生一百个随机数据点作为训练数据
Number = 100
x = np.linspace(0, Number, Number)
y = 3 * np.linspace(0, Number, Number) + np.random.uniform(-12, 12, Number)

# 接下来创建模型：
def create_model():
    # 创建 Sequential 模型
    model = Sequential()
    # 使用具有线性激活函数的全连接层
    model.add(Dense(input_dim=1, units=1, activation='linear', kernel_initializer='uniform'))
    # 使用均方差(mse)作为损失函数，Adam作为优化器编译模型 
    model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=0.1))
    return model
linear_reg_model = create_model()

# 编译模型完成后，就可以使用model.fit() 方法使用训练数据的训练模型：
linear_reg_model.fit(x, y, epochs=2000, validation_split=0.2, verbose=2)

# 训练后，就可以获得可学习参数 w 和 b，这些值将用于接下来的：
def get_weights(model):
    w = model.get_weights()[0][0][0]
    b = model.get_weights()[1][0]
    return w, b
w_final, b_final = get_weights(linear_reg_model)

# 接下来，我们可以用以下方式进行预测：
predictions = w_final * x + b_final

# 还可以保存模型：
linear_reg_model.save_weights("../opencvStudy/test67/my_model.h5")


# 接下来，我们可以加载预先训练的模型来进行预测：
# 加载模型
linear_reg_model.load_weights('../opencvStudy/test67/my_model.h5')
# 构建测试数据集
M = 3
new_x = np.linspace(Number + 1, Number + 10, M)
# 使用模型进行预测
new_predictions = linear_reg_model.predict(new_x)

# 最后可以将 训练数据集、训练完成后的线性模型 以及 预测数据 进行可视化：
plt.subplot(1, 3, 1)
plt.plot(x, y, 'ro', label='Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

plt.subplot(1, 3, 2)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result', fontsize=10)

plt.subplot(1, 3, 3)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.plot(new_x, new_predictions, 'ro', label='New predicted data', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicting new points', fontsize=10)

plt.legend()
plt.show()