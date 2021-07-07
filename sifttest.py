import cv2
import numpy as np
#import pdb
#pdb.set_trace()#turn on the pdb prompt
 
#read image
img01 = cv2.imread('/home/lzq/图片/01.png',cv2.IMREAD_COLOR)
gray01 = cv2.cvtColor(img01,cv2.COLOR_BGR2GRAY)

img02 = cv2.imread('/home/lzq/图片/02.png',cv2.IMREAD_COLOR)
gray02 = cv2.cvtColor(img02,cv2.COLOR_BGR2GRAY)

# cv2.imshow('origin',img)
 
#SIFT
detector = cv2.xfeatures2d.SIFT_create()
keypoints01 = detector.detect(gray01,None)
keypoints02 = detector.detect(gray02,None)

cv2.drawKeypoints(gray01,keypoints01,img01)
cv2.drawKeypoints(gray02,keypoints02,img02)



cv2.imshow('test01',img01)
cv2.imshow('test02',img02)
cv2.waitKey(0)
cv2.destroyAllWindows()


