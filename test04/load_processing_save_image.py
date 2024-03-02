# 计算机视觉项目处理流程示例

# 三个步骤：加载、处理和保存

import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("path_image_input", help="Path to input image to be displayed")
parser.add_argument("path_image_output", help="Path of the processed image to be saved")
args = vars(parser.parse_args())
image_input = cv2.imread(args["path_image_input"])
cv2.imshow("loaded image", image_input)
gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) # 注：我们默认的是加载BGR彩色图像，若是加载RGB彩色图像，并且想要将其转换成灰度图像，应该使用：cv2.COLOR_RGB2GRAY
cv2.imshow("gray image", gray_image)
cv2.imwrite(args["path_image_output"], gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

#终端输入：python test04/load_processing_save_image.py ../opencvStudy/test04/imgs/test01.jpeg test04/gray_image.png