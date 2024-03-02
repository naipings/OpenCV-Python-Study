# 使用OpenCV写入图像

import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("path_image", help="Path to input image to be displayed")
parser.add_argument("path_image_output", help="Path of the processed image to be saved")
args = parser.parse_args()
image = cv2.imread(args.path_image)
args = vars(parser.parse_args())
cv2.imwrite(args["path_image_output"],image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 在终端运行，输入命令：python test04/write_image.py ../opencvStudy/test04/imgs/test01.jpeg test04/copy_image.jpg