# 一些基于矩的对象特征
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 我们已经知道了，矩是从轮廓计算的特征，虽然其没有直接理解表征其几何含义，但可以根据矩计算一些几何特性。
# 接下来，我们首先计算检测到的轮廓的矩，然后，据此计算一些对象特征：
M = cv2.moments(contours[0])
print("Contour area: '{}'".format(cv2.contourArea(contours[0])))
print("Contour area: '{}'".format(M['m00']))

# 矩m_00给出了轮廓的区域，这等价于函数cv2.contourArea()。要计算轮廓的质心，需要使用以下方法：
print("center X : '{}'".format(round(M['m10'] / M['m00'])))
print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

# 圆度k是测量轮廓接近完美圆轮廓的程度，轮廓圆度计算公式如下：
# k = P^2 / (4*A*Pi)
# 其中，P是轮廓的周长，A是轮廓的区域面积，Pi是圆周率。
# 如果轮廓为圆形，其圆度为1；k值越高，它将越不像圆：
def roundness(contour, moments):
    """计算轮廓圆度"""
    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k

# 偏心率（也称为伸长率）是一种衡量轮廓伸长的程度。偏心ε可以直接从对象的长半轴a和短半轴b计算得出：
#  ε = √((a^2 - b^2) / b^2)
# 因此，计算轮廓的偏心度的一种方法是首先计算拟合轮廓的椭圆，然后计算出椭圆的导出a和b；最后，利用上述公式计算ε：
def eccentricity_from_ellipse(contour):
    """利用拟合的椭圆计算偏心率"""
    # 拟合椭圆
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc

# 另一种方法是通过使用轮廓矩来计算偏心率：（公式见：教程网址，或者截屏图片）
def eccentricity_from_moments(moments):
    """利用轮廓矩计算偏心率"""

    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc

# 纵横比是轮廓边界矩形的宽度与高度的比率，可以基于cv2.boundingRect()计算的最小边界矩形的尺寸来计算纵横比：
def aspect_ratio(contour):
    """计算纵横比"""

    x, y, w, h = cv2.boundingRect(contour)
    res = float(w) / h
    return res


# 需要注意的是：为了更精确的描述复杂对象，应该使用高阶矩或者更复杂的矩，对象越复杂，为了最大限度地减少从矩重构对象的误差，应计算的矩阶越高。