# -*- coding:utf-8 -*-
import cv2
import numpy as np
import dlib
import time
import glob as gb
import datetime

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"  
  
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(PREDICTOR_PATH)  

img_path = gb.glob("./test/*.jpg")
n = 1
for path in img_path:
	img = cv2.imread(path)
	rows,cols,dim=img.shape
        start = time.time()
	rects = detector(img,1)
	for (i, rect) in enumerate(rects):
		landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()]) 	
               	end = time.time() -start
		cv2.rectangle(img,(int(rect.left()),int(rect.top())),(int(rect.right()),int(rect.bottom())),(0,0,255),2)
       	        for idx,point in enumerate(landmarks):  
                	pos = (point[0,0],point[0,1])  
        		#cv2.putText(im,str(idx),pos,  
                	#fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  
                        #fontScale=0.4,  
			#color=(0,0,255))  
        		#6.绘制特征点  
        		cv2.circle(img,pos,3,color=(0,255,0))  
	print ("图片 %s 分辨率大小为: %d*%d，检测到 %d 个人脸。总耗时 %f ms,平均每个人脸耗时 %f ms" %(path[-5:],rows,cols,len(rects),end*1000,end*1000/len(rects)))
	s = "dlib_%d.jpg" %(n)
        cv2.imwrite(s,img)
	n += 1
	cv2.namedWindow("im",2)  
	cv2.imshow("im",img)  
	cv2.waitKey(0)  
