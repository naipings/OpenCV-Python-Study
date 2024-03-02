# 灰度图像访问和操作OpenCV中的像素

import cv2
# 灰度图像只有一个通道，因此，在处理时会引入一些差异。

# 使用cv2.imread()读取图像时，需要第二个参数，因为我们希望以灰度加载图像。第二个参数是一个标志位，指定读取图像的方式。
# 以灰度加载图像所需的值是 cv2.IMREAD_GRAYSCALE
gray_img = cv2.imread('../opencvStudy/test03/imgs/test01.jpeg', cv2.IMREAD_GRAYSCALE)

# 如果我们打印图像的尺寸（使用gray_img.shape），只能得到两个值，即行和列。在灰度图像中，不提供通道信息：
dimensions = gray_img.shape
# shape以元组形式返回图像的维度

# 在灰度图像中，只能获得一个值（通常成为像素的强度）。例如：如果我们想得到像素（x=40，y=6）处的像素强度：
# i = gray_img[6, 40]

# 图像的像素值也可以以相同的方式修改。例如：如果要将像素（x=40，y=6）处的值更改为黑色（强度等于0）
# gray_img[6, 40] = 0

cv2.imshow("gray image", cv2.resize(gray_img,None,fx=0.1,fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()