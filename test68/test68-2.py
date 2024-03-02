#  cv2.dnn.blobFromImages() 函数详解
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# cv2.dnn.blobFromImages() 函数介绍见test68.py

# 接下来，测试 cv2.dnn.blobFromImages() 函数:

# 我们首先加载目标文件夹中的所有图像，然后使用 cv2.dnn.blobFromImages() 函数创建一个四维blob，
# 同样，我们编写 get_images_from_blob() 函数用于执行逆预处理变换以再次获取输入图像。
def get_images_from_blob(blob_imgs, scalefactor, dim, mean, swap_rb, mean_added):
    images_from_blob = cv2.dnn.imagesFromBlob(blob_imgs)
    imgs = []

    for image_blob in images_from_blob:
        image_from_blob = np.reshape(image_blob, dim) / scalefactor
        image_from_blob_mean = np.uint8(image_from_blob)
        image_from_blob = image_from_blob_mean + np.uint8(mean)
        if mean_added is True:
            if swap_rb:
                image_from_blob = image_from_blob[:, :, ::-1]
            imgs.append(image_from_blob)
        else:
            if swap_rb:
                image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
            imgs.append(image_from_blob_mean)
    return imgs

# 加载图像并构造图像列表
images = []
for img in glob.glob('../opencvStudy/test68/68-2imgs/*.jpg'):
    images.append(cv2.imread(img))
# 调用 cv2.dnn.blobFromImages() 函数
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
# 打印形状
print(blob_images.shape)
# 从 blob 中获取不同的图像
# imgs_from_blob 图像对应于调整大小为 (300,300) 的原始 BGR 图像，并且已经添加了通道均值
imgs_from_blob = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
# img_from_blob_swap 图像对应于调整大小为 (300,300) 的原始 RGB 图像
# img_from_blob_swap 交换了蓝色和红色通道，并且已经添加了通道均值
imgs_from_blob_swap = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], True, True)
# img_from_blob_mean 图像对应于调整大小为 (300,300) 的原始 BGR 图像，其并未添加通道均值
imgs_from_blob_mean = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], False, False)
# img_from_blob_mean_swap 图像对应于调整为 (300,300) 的原始 RGB 图像
# img_from_blob_mean_swap 交换了蓝色和红色通道，并未添加通道均值
imgs_from_blob_mean_swap = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], True, False)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=10)
    plt.axis('off')

for i in range(len(images)):
    show_img_with_matplotlib(imgs_from_blob[i], "img from blob " + str(imgs_from_blob[i].shape), i * 4 + 1)
    show_img_with_matplotlib(imgs_from_blob_swap[i], "img from blob swap " + str(imgs_from_blob_swap[i].shape), i * 4 + 2)
    show_img_with_matplotlib(imgs_from_blob_mean[i], "img from blob mean " + str(imgs_from_blob_mean[i].shape), i * 4 + 3)
    show_img_with_matplotlib(imgs_from_blob_mean_swap[i], "img from blob mean swap " + str(imgs_from_blob_mean_swap[i].shape), i * 4 + 4)

plt.show()