import cv2
import numpy as np
from djitellopy import Tello

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)



# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0
print(me.get_battery())
me.streamoff()
me.streamon()
startCounter =0




deadZone=100

knn = cv2.ml.KNearest_load('./mnist_knn_test.xml')   # 載入模型

global imgContour

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",19,179,empty)
cv2.createTrackbar("HUE Max","HSV",35,179,empty)
cv2.createTrackbar("SAT Min","HSV",107,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",89,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",166,255,empty)
cv2.createTrackbar("Threshold2","Parameters",171,255,empty)
cv2.createTrackbar("Area","Parameters",3750,30000,empty)

def getContours(img, imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

while True:
    # _, img = cap.read()
    #tello
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (frameWidth, frameHeight))
    x, y, w, h = 0, 0, 640, 480           # 定義擷取數字的區域位置和大小
    img_num = img.copy()                     # 複製一個影像作為辨識使用
    img_num = img_num[y:y+h, x:x+w]          # 擷取辨識的區域

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)    # 針對白色文字，做二值化黑白轉換，轉成黑底白字

    img_num = cv2.resize(img_num, (28, 28))  # 縮小成 28x28，和訓練模型對照
    img_num = img_num.astype(np.float32)  # 轉換格式
    img_num = img_num.reshape(-1, )  # 打散成一維陣列資料，轉換成辨識使用的格式
    img_num = img_num.reshape(1, -1)
    img_num = img_num / 255
    img_pre = knn.predict(img_num)  # 進行辨識
    num = str(int(img_pre[1][0][0]))  # 取得辨識結果

    text = num  # 印出的文字內容
    org = (320, 240)  # 印出的文字位置
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出的文字字體
    fontScale = 2  # 印出的文字大小
    color = (0, 0, 255)  # 印出的文字顏色
    thickness = 2  # 印出的文字邊框粗細
    lineType = cv2.LINE_AA  # 印出的文字邊框樣式
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)  # 印出文字

    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower = np.array([101,50,38])
    upper = np.array([110,255,255])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)


    #tello
    if startCounter == 0:
        me.takeoff()
        startCounter = 1

    if cv2.waitKey(1) & 0xFF == ord('a'):
        me.left_right_velocity = -120
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        me.left_right_velocity = 120
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        me.up_down_velocity = 120
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        me.up_down_velocity = -120
    elif cv2.waitKey(1) & 0xFF == ord('i'):
        me.for_back_velocity = 120
    elif cv2.waitKey(1) & 0xFF == ord('k'):
        me.for_back_velocity = -120
    else:
        me.left_right_velocity = int(me.left_right_velocity/2);
        me.for_back_velocity = int(me.for_back_velocity/2);
        me.up_down_velocity = int(me.up_down_velocity/2);
        me.yaw_velocity = int(me.yaw_velocity/2)
    if me.send_rc_control:
        me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)

    cv2.imshow('Result', imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

cap.release()
cv2.destroyAllWindows()