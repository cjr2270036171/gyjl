
import numpy as np
import cv2
import imutils

def dovision(img,b_d,b_sigmaColor,b_sigmaSpace,threshold1,threshold2):
    image = imutils.resize(img, height=800)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, b_d, b_sigmaColor, b_sigmaSpace)
    edged = cv2.Canny(gray,float(threshold1),float(threshold2))
    cv2.imwrite("./demo1.jpg", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def v(image):
    coords = [(1484.0, 1182.0), (2271.0, 1191.0), (2219.0, 1772.0), (1569.0, 1748.0)]
    image = cv2.imread(image)
    image = imutils.resize(image,height=800)
    pts = np.array(coords, dtype="float32")
    warped = four_point_transform(image, pts)
    dovision(warped, 5,10,10,17,14)
    cv2.waitKey(0)
def haha(j,i):
    a=cv2.floodFill(copyIma, mask, (j, i), (0, 0,0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    if a[0]>0:
        list.append(a[0])
def vision(hr,key=1):
    v(hr)
    global list
    list = []
    global src
    src = cv2.imread("./demo1.jpg")
    global copyIma
    copyIma = src.copy()
    global h
    global w
    h, w = src.shape[:2]
    global mask
    mask = np.zeros([h + 2, w + 2], np.uint8)
    global gray
    gray = cv2.cvtColor(copyIma, cv2.COLOR_BGR2GRAY)
    # 如果参数为1，返回面积，如果为2，返回方差
    if key==1:
        for i in range(0, h, 1):
            for j in range(0, w, 1):
                if gray[i, j] == 255:
                    haha(j, i)
    elif key==2:
        arr = cv2.imread("./demo1.jpg", 0)
        arr_mean = np.mean(arr)
        arr_var = np.var(arr)
        arr_std = np.std(arr, ddof=1)
        list=[round(arr_mean,3),round(arr_var,3),round(arr_std,3)]
    elif key==3:
        for i in range(0, h, 1):
            for j in range(0, w, 1):
                if gray[i, j] == 255:
                    haha(j, i)
        arr = list
        arr_mean = np.mean(arr)
        arr_var = np.var(arr)
        arr_std = np.std(arr, ddof=1)
        list=[round(arr_mean,3),round(arr_var,3),round(arr_std,3)]
    return list
#第一个参数为图片地址，第二个参数可选：1:返回面积；2：根据图片计算[平均值,方差,标准差]；3：根据面积计算[平均值,方差,标准差]
#例：
# ll=vision("./01.jpg",1)
# print(ll)


