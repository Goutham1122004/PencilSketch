import numpy as np
import cv2

highThresh = 0.4
lowThresh = 0.1
imgFileLst = ('./sam1.jpg', './sam2.jpg')

def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(opImgx, opImgy)

def sketch(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 1, edgImg1, 1, 0)
    opImg = 255 - edgImg
    return opImg

if __name__ == '__main__':
    for imgFile in imgFileLst:
        print(imgFile)
        img = cv2.imread(imgFile, 0)
        opImg = sketch(img)
        window_name = imgFile
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 600)
        cv2.imshow(window_name, opImg)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
