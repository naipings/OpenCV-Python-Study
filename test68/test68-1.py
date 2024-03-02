# cv2.dnn.blobFromImage() 函数详解
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# cv2.dnn.blobFromImage() 函数介绍见test68.py

# 为了测试 cv2.dnn.blobFromImage() 函数，首先加载一个BGR图像，然后使用 cv2.dnn.blobFromImage() 函数创建一个四维blob。
# 然后，我们编写 get_image_from_blob() 函数，该函数可用于执行逆预处理变换以再次获取输入图像，以更好的理解 cv2.dnn.blobFromImage() 函数的预处理：
# 加载图像
image = cv2.imread("../opencvStudy/test68/imgs/test2.jpg")
# 调用 cv2.dnn.blobFromImage() 函数
blob_image = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
# blob_image 的尺寸为 (1, 3, 300, 300)
print(blob_image.shape)

def get_image_from_blob(blob_img, scalefactor, dim, mean, swap_rb, mean_added):
    images_from_blob = cv2.dnn.imagesFromBlob(blob_img)
    image_from_blob = np.reshape(images_from_blob[0], dim) / scalefactor
    image_from_blob_mean = np.uint8(image_from_blob)
    image_from_blob = image_from_blob_mean + np.uint8(mean)

    if mean_added is True:
        if swap_rb:
            image_from_blob = image_from_blob[:, :, ::-1]
        return image_from_blob
    else:
        if swap_rb:
            image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
        return image_from_blob_mean

# 从 blob 中获取不同的图像
# img_from_blob 图像对应于调整为 (300,300) 的原始 BGR 图像，并且已经添加了通道均值
img_from_blob = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
# img_from_blob_swap 图像对应于调整大小为 (300,300) 的原始 RGB 图像
# img_from_blob_swap 交换了蓝色和红色通道，并且已经添加了通道均值
img_from_blob_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, True)
# img_from_blob_mean 图像对应于调整大小为 (300,300) 的原始 BGR 图像，其并未添加通道均值
img_from_blob_mean = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, False)
# img_from_blob_mean_swap 图像对应于调整为 (300,300) 的原始 RGB 图像
# img_from_blob_mean_swap 交换了蓝色和红色通道，并未添加通道均值
img_from_blob_mean_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, False)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=10)
    plt.axis('off')

show_img_with_matplotlib(img_from_blob, "img from blob " + str(img_from_blob.shape), 1)
show_img_with_matplotlib(img_from_blob_swap, "img from blob swap " + str(img_from_blob.shape), 2)
show_img_with_matplotlib(img_from_blob_mean, "img from blob mean " + str(img_from_blob.shape), 3)
show_img_with_matplotlib(img_from_blob_mean_swap, "img from blob mean swap " + str(img_from_blob.shape), 4)

plt.show()