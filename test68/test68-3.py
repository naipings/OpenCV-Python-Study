# cv2.dnn.blobFromImage() 和 cv2.dnn.blobFromImages() 的crop参数
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# crop参数介绍见test68.py

def get_cropped_img(img):
    img_copy = img.copy()
    size = min(img_copy.shape[1], img_copy.shape[0])
    x1 = int(0.5 * (img_copy.shape[1] - size))
    y1 = int(0.5 * (img_copy.shape[0] - size))
    return img_copy[y1:(y1 + size), x1:(x1 + size)]

# get_images_from_blob() 函数用于执行逆预处理变换以再次获取输入图像
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

images = []
for img in glob.glob('../opencvStudy/test68/68-3imgs/*.jpg'):
    images.append(cv2.imread(img))
# 使用 get_cropped_img() 函数进行裁剪
cropped_img = get_cropped_img(images[0])
# crop = False 不进行裁剪
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
print(blob_images)
# crop = True 进行裁剪
blob_blob_images_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)
imgs_from_blob = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
imgs_from_blob_cropped = get_images_from_blob(blob_blob_images_cropped, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=10)
    plt.axis('off')

for i in range(len(images)):
    show_img_with_matplotlib(imgs_from_blob[i], "img {} from blob ".format(i) + str(imgs_from_blob[i].shape), i + 1)
    show_img_with_matplotlib(imgs_from_blob_cropped[i], "img {} from blob cropped ".format(i) + str(imgs_from_blob[i].shape), i + 5)

plt.show()