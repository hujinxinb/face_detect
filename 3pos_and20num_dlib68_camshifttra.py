'''
Main program
@Author: David Vu and yc

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
from trackers import CamshiftTracker,CorrelationTracker
from align_custom_1 import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import dlib
import time

time1 = time.time()

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"  
  
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(PREDICTOR_PATH)  
t = time.time() - time1
print (t)
def main(args):
    mode = args.mode
    if(mode == "camera"):
	while True:
       		img,face_box,recog_data,flage = camera_recog()
		if flage:
			track(img,face_box,recog_data)


    elif mode == "input":
        create_manual_data()
    else:
        raise ValueError("Unimplemented mode")
'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them 
    -> each cropped face is categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions. 
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)
    
'''
def camera_recog():
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0); #get input from webcam
    c = 0
    time_start = time.time()
    flage = True
    while c < 2:
	time_start = time.time()
        _,frame = vs.read();
	c += 1	
	print (c)
        #u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
	rects = detector(frame,1)
#	rects, landmarks = face_detect.detect_face(frame,20);#min face size is set to 80x80
       	aligns = []
       	positions = []
	if rects:
	       	for (i, rect) in enumerate(rects):	
			landmarks = np.matrix([[p.x,p.y] for p in predictor(frame,rects[i]).parts()])
                        aligned_face, face_pos = al(160,frame,landmarks)
			#aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
          		aligns.append(aligned_face)
           		positions.append(face_pos)
	features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions)
	
	if rects:
                for (i, rect) in enumerate(rects):
			cv2.rectangle(frame,(int(rect.left()),int(rect.top())),(int(rect.right()),int(rect.bottom())),(0,0,255),2)
        	       	cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect.left(),rect.top()),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
	t = time.time() - time_start
	fps = ("%.2f" % (1.0/t))
	cv2.putText(frame,"each frame det and recog:"  + str(fps),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
	#cv2.namedWindow("Image", cv2.INDOW_AUTOSIZE)
        cv2.imshow("Image",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
    print (type(rects))
    if not len(rects):
	flage = False
    return frame,rects,recog_data,flage

'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''
def findPeople(features_arr, positions, thres = 0.7, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D_1.txt','r')
    data_set = json.loads(f.read());
    returnRes = []
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes

'''
Description:
User input his/her name or ID -> Images from Video Capture -> detect the face -> crop the face and align it 
    -> face is then categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Append each newly extracted face 128D vector to its corresponding position type (Center, Left, Right)
    -> Press Q to stop capturing
    -> Find the center ( the mean) of those 128D vectors in each category. ( np.mean(...) )
    -> Save
    
'''
def create_manual_data():
    vs = cv2.VideoCapture(0); #get input from webcam
    print("Please input new user ID:")
    new_name = raw_input("input ID:"); #ez python input()

    print ("ce shi dai ma")
    #f = open('./facerec_128D.txt','r');
    #data_set = json.loads(f.read());
    data_set = {}
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    person_features = {"Left" : [], "Right": [], "Center": []}
    pic_num = {"Left" : [], "Right": [], "Center": []}
    mark = [None]*10
    num = [0,0,0]
    print (type(mark),mark)
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read()
        rects = detector(frame,1)   # min face size is set to 20x20
	if rects:
		for (i, rect) in enumerate(rects):		
			landmarks = np.matrix([[p.x,p.y] for p in predictor(frame,rects[i]).parts()])  
			#aligned_frame, pos = aligner.align(160,frame,mark)
			aligned_frame, pos = al(160,frame,landmarks)
			print (pos)
			if num[0] < 20:
                        	print ("Please put your head to Left")
                        	if pos == "Left":
                                	pic_num[pos] = num[0]
                                	num[0] += 1
                                	person_imgs[pos].append(aligned_frame)
					print (pos,num[0])

                    	if num[1] < 20:
                        	print ("Please put your head to Right")
                        	if pos == "Right":
                                	pic_num[pos] = num[1]
                                	num[1] += 1
                                	person_imgs[pos].append(aligned_frame)
					print (pos,num[1])
					
                    	if num[2] < 20:
                        	print ("Please put your head to Center")
                        	if pos == "Center":
                                	pic_num[pos] = num[2]
                                	num[2] += 1
                                	person_imgs[pos].append(aligned_frame)
					print (pos,num[2])

			#cv2.imshow("Image",aligned_frame)
			cv2.rectangle(frame,(int(rect.left()),int(rect.top())),(int(rect.right()),int(rect.bottom())),(0,0,255),2)
			for idx,point in enumerate(landmarks): 
			        position = (point[0,0],point[0,1]) 
			        #cv2.putText(frame,str(idx),pos,  
                    		#	fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  
                    		#	fontScale=0.4,  
  
                		#	color=(0,0,255))  
			        cv2.circle(frame,position,3,color=(0,255,0))  

        cv2.imshow("Captured face", frame)
	cv2.waitKey (1)
	
	if (num[0]==num[1]==num[2]==20):
                break
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord("q"):
        #    break
    
    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    print ("done done done done")
    f = open('./facerec_128D_dlib68.txt', 'w');
    f.write(json.dumps(data_set))

def track(frame,face_box,recog_data):
    n = 0
    trackers = {"camshift":CamshiftTracker,"correlation":CorrelationTracker}
    tracktool = trackers["camshift"]
    objTracker = []
    facenum = 0
    for i,rect in enumerate(face_box):
        bs = [(rect.left(), rect.top()),(rect.right(),rect.bottom())]
        objTracker.append(tracktool(bs))
        facenum = facenum + 1
    cap = cv2.VideoCapture(0)
    while n < 10000:
        time_start = time.time()
        ret, img = cap.read()
        for i in range(0,facenum):
            trackPts = objTracker[i].track(img)
            (x,y,w,h) = trackPts
            t = time.time() - time_start
            fps = ("%.2f" % (1.0 / t))
            cv2.putText(img, "Only Track FPS:" + str(fps), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
            cv2.rectangle(img, (x,y),(x+w,y+h), (255, 255, 255), 1)
            print ("Object {} tracked at [{}, {}] \r".format(i, (x,y),(x+w,y+h)))
            cv2.putText(img, recog_data[i][0] + " - " + str(recog_data[i][1]) + "%",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img)
            n += 1
            print (n)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def al(size, frame, landmarks):
    
    mark = [None]*10
    x1 = landmarks[36]
    x2 = landmarks[45]
    x3 = landmarks[30]
    x4 = landmarks[48]
    x5 = landmarks[54]
    mark[0] = x1[0,0]
    mark[1] = x2[0,0]
    mark[2] = x3[0,0]
    mark[3] = x4[0,0]
    mark[4] = x5[0,0]
    mark[5] = x1[0,1]
    mark[6] = x2[0,1]
    mark[7] = x3[0,1]
    mark[8] = x4[0,1]
    mark[9] = x5[0,1]	

    mark = np.array(mark)
    print (type(mark),mark.shape)
    aligned_frame, pos = aligner.align(160,frame,mark)
    return aligned_frame, pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main(args);
