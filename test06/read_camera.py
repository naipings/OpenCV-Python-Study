# 访问捕获画面对象的属性

import cv2

capture = cv2.VideoCapture(0)

# 获取 VideoCapture 的属性 (frame width, frame height and frames per second (fps)):
# 使用capture.get(property_identifier)访问capture对象的某些属性，例如：帧宽度、帧高度和每秒帧数。如果调用不受支持的属性，则返回值将为0
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# 打印属性值
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

# Check if camera opened successfully
if capture.isOpened()is False:
    print("Error opening the camera")
    
while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        cv2.imshow('Input frame from the camera', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
        
capture.release()
cv2.destroyAllWindows()

# 直接 Run Code就行