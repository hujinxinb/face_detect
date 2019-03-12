
import cv2
import mtcnn_detect
from mtcnn_detect import MTCNNDetect


face_detect = MTCNNDetect(); #scale_factor, rescales image for faster detection
    

rects, landmarks = face_detect.detect_face(frame, 20);  # min face size is set to 20x20

cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
for j in range(0,5):
     	cv2.circle(frame,(int(landmarks[i][j]),int(landmarks[i][j+5])),2,(0,0,255))
	cv2.namedWindow("Image")
        cv2.imshow("Image", frame)
	cv2.waitKey (1) 

