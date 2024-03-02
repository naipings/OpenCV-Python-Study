# 在OpenCV中读取图像

import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("path_image", help="Path to input image to be displayed")
args = parser.parse_args()
image = cv2.imread(args.path_image)
args = vars(parser.parse_args())
image2 = cv2.imread(args["path_image"])

cv2.imshow("loaded image",image)

cv2.imshow("loaded image2",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 执行程序时，在终端输入：python test04/load_image.py ../opencvStudy/test04/imgs/test01.jpeg