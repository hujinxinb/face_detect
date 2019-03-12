'''
Main program
@Author: David Vu

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import dlib
import time

def main(args):
    mode = args.mode
    if(mode == "camera"):
	while True:
	       	img,face_box,recog_data,flage = camera_recog()
		#if flage:
		#	track(img,face_box,recog_data)


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
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480)) 
    
    c = 0
    time_start = time.time()
    flage = True
	
    while True:
	time_start = time.time()
        _,frame = vs.read();
	if (c %1==0 or c == 0):
		rects, landmarks = face_detect.detect_face(frame,20);#min face size is set to 80x80
       		aligns = []
       		positions = []
       		for (i, rect) in enumerate(rects):
       			aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
        		aligns.append(aligned_face)
        		positions.append(face_pos)
		features_arr = extract_feature.get_features(aligns)
       		recog_data = findPeople(features_arr,positions)
		for (i,rect) in enumerate(rects):
			print (len(recog_data))
        		cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
        	 	cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
		t = time.time() - time_start
		fps = ("%.2f" % (1.0/t))
	c += 1
#	cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
#	cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

#	cv2.putText(frame,"each 3frame det and recog:"  + str(fps),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
  	out.write(frame) 
	cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Image",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
#    print (type(rects))
    out.release()  
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
def findPeople(features_arr, positions, thres = 0.5, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D_mtcnn5.txt','r')
    print ("load load ")
    data_set = json.loads(f.read());
    print (len(data_set))
    returnRes = [];
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
    person_imgs = {"Left" : [], "Right": [], "Center": [],"Down": [],"Up": []}
    print (type(person_imgs))
    person_features = {"Left" : [], "Right": [], "Center": [],"Down": [],"Up": []}
    pic_num = {"Left" : [], "Right": [], "Center": [],"Down": [],"Up": []}
    num = [0,0,0,0,0]
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
	
        rects, landmarks = face_detect.detect_face(frame, 20);  # min face size is set to 20x20
	print (type(landmarks),landmarks.shape)
	if rects:
		for (i, rect) in enumerate(rects):
        	    aligned_frame, pos = aligner.align(160,frame,landmarks[i])
		    print (pos)
		    if num[0] < 20: 
			print ("Please put your head to Left")
			if pos == "Left":    	
				pic_num[pos] = num[0]
				num[0] += 1
				person_imgs[pos].append(aligned_frame)

		    if num[1] < 20:        
                        print ("Please put your head to Right")
                        if pos == "Right":
                                pic_num[pos] = num[1]
                                num[1] += 1
                                person_imgs[pos].append(aligned_frame)

		    if num[2] < 20:        
                        print ("Please put your head to Center")
                        if pos == "Center":
                                pic_num[pos] = num[2]
                                num[2] += 1
                                person_imgs[pos].append(aligned_frame)

		    if num[3] < 20:        
                        print ("Please put your head to Down")
                        if pos == "Down":
                                pic_num[pos] = num[3]
                                num[3] += 1
                                person_imgs[pos].append(aligned_frame)
		
   		    if num[4] < 20:        
                        print ("Please put your head to Up")
                        if pos == "Up":
                                pic_num[pos] = num[4]
                                num[4] += 1
                                person_imgs[pos].append(aligned_frame)

		    cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
		    for j in range(0,5):
	        	cv2.circle(frame,(int(landmarks[i][j]),int(landmarks[i][j+5])),2,(0,0,255))
	print (num[0],num[1],num[2],num[3],num[4])
	cv2.namedWindow("Image",cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Image", frame)
	cv2.waitKey (1) 
	if (num[0]==num[1]==num[2]==num[3]==num[4]==20):
		break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
	person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    print (person_features)
    data_set[new_name] = person_features;
    print ("done done done done")
    f = open('./facerec_128D_mtcnn5.txt', 'a+');
    f.write(json.dumps(data_set))
    f.write('\n')

def track(frame,face_box,recog_data):
    
    n = 0
    # Initial co-ordinates of the object to be tracked 
    # Create the tracker object
    tracker = [dlib.correlation_tracker() for _ in xrange(len(face_box))]
    # Provide the tracker the initial position of the object
    for i, rect in enumerate(face_box):
	
	rect = list(rect)
	#print (rect[0],rect[1],rect[2],rect[3])
	#print (type(rect))
	tracker[i].start_track(frame, dlib.rectangle(rect[0],rect[1],(rect[0] + rect[2]),(rect[1]+rect[3])))
	#print (rect)

    cap = cv2.VideoCapture(0)
    while n < 100:
	time_start = time.time()
	ret,img = cap.read()
    	# Update the tracker  
	for i in xrange(len(tracker)):
		tracker[i].update(img)
		
       		# Get the position of the object, draw a 
       		# bounding box around it and display it.
      		rect = tracker[i].get_position()
		
       		pt1 = (int(rect.left()), int(rect.top()))
       		pt2 = (int(rect.right()), int(rect.bottom()))
		
		t = time.time()-time_start
		fps = ("%.2f" % (1.0/t))

		cv2.putText(img,"only track" + str(fps),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)	

       		cv2.rectangle(img, pt1, pt2, (255, 255, 255), 1)
		print ("Object {} tracked at [{}, {}] \r".format(i, pt1, pt2))
		cv2.putText(img,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        	cv2.imshow("Image", img)
		n += 1
		print (n)
		# Continue until the user presses ESC key
		if cv2.waitKey(1) & 0xFF == ord('q'):
	                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main(args);
