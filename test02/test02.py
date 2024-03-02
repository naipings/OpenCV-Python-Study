# 在不同颜色空间中访问和操作OpenCV中的像素

import cv2
img = cv2.imread('../opencvStudy/test02/imgs/test01.jpeg')
# 加载图片过后，获取图片的一些属性：
# shape，它将告诉我们行、列和通道的数量（如果图像是彩色的）
dimensions = img.shape
# 图片的大小（img.size = 图像高度*图像宽度*图像通道数）
total_number_of_elements = img.size
# 图像的数据类型，可以通过img.dtype获得。因为像素值在[0,255]范围内，所以图像数据类型是uint8（unsigned char）
image_dtype = img.dtype

# 使用cv2.imshow()显示图像时，窗口会自动适应图像大小。此函数的第一个参数是窗口名，第二个参数是要显示的图像。
cv2.imshow("original image", img)

# 键盘绑定函数————cv2.waitkey()，它为任何键盘事件等待指令的毫秒数。参数是以毫秒为单位的时间。
# 当执行到此函数时，程序将暂停执行，当按下任何键后，程序将继续执行。如果毫秒数为0（cv2.waitkey(0)），它将无限期地等待键盘敲击事件：
cv2.waitKey(0)

# 访问（读取）某个像素值，我们需要向img变量（包含加载的图像）提供所需像素的行和列，例如，要获得（x=40，y=6）处的像素值：
# (b, g, r) = img[6, 40]
# 我们在三个变量(b, g, r)中存储了三个像素值。请牢记OpenCV对彩色图像使用BGR格式。
# 另外，我们可以一次仅访问一个通道。在本例中，我们将使用所需通道的行、列和索引进行索引。例如，要仅获取像素（x=40，y=6）处的蓝色值：
# b = img[6, 40, 0]

# 像素值也可以以相同的方式进行修改。例如，要将像素（x=40，y=6）处设置为红色：
# img[6, 40] = (0 ,0, 255)

# 有时，需要处理某个区域而不是一个像素。在这种情况下，应该提供值的范围（也称切片），而不是单个值。例如：要获取图像的左上角：
# top_left_corcer = img[0:50, 0:50]
# 变量top_left_corcer可以看做是另一个图像（比img小），但是我们可以用同样的方法处理它

# 最后，如果想要关闭并释放所有窗口，需要使用cv2.destroyAllWindows()函数：
cv2.destroyAllWindows()