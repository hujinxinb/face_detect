#coding=utf-8

# Form implementation generated from reading ui file 'ui_1.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


import numpy as np

import tensorflow as tf
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

import time
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import random
import time
import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np



import sklearn

from sklearn.externals import joblib


#
# #face detection parameters
# minsize = 20 # minimum size of face
# threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
# factor = 0.709 # scale factor
#
# #facenet embedding parameters
#
# model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
# model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
# image_size=96 #"Image size (height, width) in pixels."
# pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
# use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
# seed=42,# "Random seed."
# batch_size= None # "Number of images to process in a batch."




frame_interval=4 # frame intervals，小了很顺但不实时，大了实时但不顺


def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


def findPeople(features_arr, positions, thres=0.7, percent_thres=70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D.txt', 'r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i, features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():

            #这里有bug 需调试
            if (positions[i]=='Up' or positions[i]=='Down'):
                positions[i]='Center'


            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data - features_128D)))
                if (distance < smallest):
                    smallest = distance;
                    result = person;
        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = "Unknown"
        returnRes.append((result, percentage))
    return returnRes





class Ui_MainWindow(object):

    video_flag = True


    def addface(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
        args = parser.parse_args(sys.argv[1:]);
        FRGraph = FaceRecGraph();
        aligner = AlignCustom();
        extract_feature = FaceFeature(FRGraph)
        face_detect = MTCNNDetect(FRGraph, scale_factor=2);  # scale_factor, rescales image for faster detection



        #这个print需要界面化
        print("Please input new user ID:")
        #new_name = raw_input("input ID:");  # ez python input()
        new_name = unicode(self.lineEdit.text().toUtf8(),'utf8','ignore')


        print ("ce shi dai ma")
        f = open('./facerec_128D.txt', 'r');
        data_set = json.loads(f.read());
        person_imgs = {"Left": [], "Right": [], "Center": []};
        person_features = {"Left": [], "Right": [], "Center": []};
        print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");

        vs = cv2.VideoCapture("rtsp://admin:1234qwer@192.168.2.131/cam/realmonitor?channel=1&subtype=0")


        while True:
            _, frame = vs.read();
            rects, landmarks = face_detect.detect_face(frame, 20);  # min face size is set to 80x80
            for (i, rect) in enumerate(rects):
                aligned_frame, pos = aligner.align(182, frame, landmarks[i])
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)


            cv2.cvtColor(frame,cv2.COLOR_BGR2RGB,frame)
            img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(img)
            self.label.setPixmap(pixmap)

            key = cv2.waitKey(1) & 0xFF
            #if key == ord("q"):
             #   break
            if self.video_flag == False:
                break

        vs.release()
        cv2.destroyAllWindows()




        for pos in person_imgs:  # there r some exceptions here, but I'll just leave it as this to keep it simple
            print ("ceshi")
            print (person_imgs[pos])
            person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]
        data_set[new_name] = person_features;



        print ("done done done done")
        f = open('./facerec_128D.txt', 'w');
        f.write(json.dumps(data_set))
        exit(0)




    def findpeopleface(self):

        self.video_flag = True

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
        args = parser.parse_args(sys.argv[1:]);
        FRGraph = FaceRecGraph();
        aligner = AlignCustom();
        extract_feature = FaceFeature(FRGraph)
        face_detect = MTCNNDetect(FRGraph, scale_factor=2);  # scale_factor, rescales image for faster detection
        print("[INFO] camera sensor warming up...")

        vs = cv2.VideoCapture('rtsp://admin:1234qwer@192.168.2.131/cam/realmonitor?channel=1&subtype=0');  # get input from webcam
        #vs = cv2.VideoCapture('test.mp4')

        c=0
        _, frame = vs.read();

        message=''

        result_dict = {}


        while True:
            timeF= frame_interval




            if (c % timeF == 0):
                # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
                rects, landmarks = face_detect.detect_face(frame, 20);  # min face size is set to 80x80
                aligns = []

                positions = []
                for (i, rect) in enumerate(rects):
                    aligned_face, face_pos = aligner.align(182, frame, landmarks[i])
                    aligns.append(aligned_face)
                    positions.append(face_pos)
                features_arr = extract_feature.get_features(aligns)

                recog_data = findPeople(features_arr, positions);

                for (i, rect) in enumerate(rects):
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                                  (0, 255, 0),2)  # draw bounding box for the face
                    cv2.putText(frame, recog_data[i][0] + " - " + str(recog_data[i][1]) + "%", (rect[0], rect[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.CV_AA)

                    if result_dict.has_key(recog_data[i][0]):
                        result_dict[recog_data[i][0]][1]+=1
                        result_dict[recog_data[i][0]][0] = (result_dict[recog_data[i][0]][0]*(result_dict[recog_data[i][0]][1]-1)+float(recog_data[i][1]))/result_dict[recog_data[i][0]][1]
                    else:
                        result_dict[recog_data[i][0]]=[float(recog_data[i][1]),1]


                cv2.cvtColor(frame,cv2.COLOR_BGR2RGB,frame)
                img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap(img)
                self.label.setPixmap(pixmap)

                #result_dict是用来保存名称和精确度的字典，将它的按值排序给result_list并定义一个字符串message来保存result_list的内容并显示message
                result_list = sorted(result_dict.items(), key=lambda item: item[1][1], reverse=True)
                message=''
                for i in result_list:
                    message+=i[0]
                    message+=': \n'
                    message+=str(i[1][0])[0:10]
                    message+='%\n'
                    message+=str(i[1][1])[0:7]
                    message+=' times\n\n'
                self.plainTextEdit.setPlainText(message)

                key = cv2.waitKey(1) & 0xFF
                if self.video_flag==False:
                    break

            _, frame = vs.read();
            c+=1

        vs.release()
        cv2.destroyAllWindows()

    def change_videoflag(self):
        if self.video_flag==True:
            self.video_flag=False


    def showVideo1(self):
        return
        # self.video_flag=True
        #
        # print('Creating networks and loading parameters')
        # gpu_memory_fraction = 0.3
        # with tf.Graph().as_default():
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #     with sess.as_default():
        #         pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
        #
        # cap = cv2.VideoCapture("rtsp://admin:1234qwer@192.168.2.131/cam/realmonitor?channel=1&subtype=0")
        #
        # #fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
        # #fps1 = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        #
        # #size1 = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        # #int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        #
        # #writer = cv2.VideoWriter("haikang1.avi", fourcc, fps1, size1)
        #
        # ret, frame = cap.read()
        # index=0
        # c = 0
        # while True:
        #
        #     timeF = frame_interval
        #
        #     #writer.write(frame)
        #
        #     path = "video"
        #     is_path_Exists = os.path.exists(path)
        #
        #     if not is_path_Exists:
        #         os.mkdir("video")
        #
        #     index+=1
        #     #python中的snprintf
        #     #outputname = "video/%d_output.jpg" % (index)
        #     #cv2.imwrite(outputname,frame)
        #
        #     if(c%timeF==0):       #frame interval==3,face detection every 3 frames
        #         find_result = []
        #
        #         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #
        #         if gray.ndim==2:
        #             img=to_rgb(gray)
        #
        #         start = time.clock()
        #
        #         bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #
        #         nrof_faces = bounding_boxes.shape[0]  # number of faces
        #         # print('The number of faces detected is{}'.format(nrof_faces))
        #
        #         for face_position in bounding_boxes:
        #             face_position = face_position.astype(int)
        #
        #             # print((int(face_position[0]), int( face_position[1])))
        #             # word_position.append((int(face_position[0]), int( face_position[1])))
        #
        #             cv2.rectangle(frame, (face_position[0],
        #                                   face_position[1]),
        #                           (face_position[2], face_position[3]),
        #                           (0, 255, 0), 2)
        #
        #         end = time.clock()
        #         time1 = end - start
        #         print time1
        #
        #
        #         cv2.cvtColor(frame,cv2.COLOR_BGR2RGB,frame)
        #         img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        #
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #
        #         if self.video_flag==False:
        #             cap.release()
        #
        #             cv2.destroyAllWindows()
        #             break
        #
        #         pixmap = QtGui.QPixmap(img)
        #
        #         self.label.setPixmap(pixmap)
        #
        #     ret,frame = cap.read()
        #     c+=1
        #
        # cap.release()
        # cv2.destroyAllWindows()

    def setUpUiStyle(self):
        #self.m_btnEnter.setStyleSheet("QPushButton{color:white;background:rgb(35,35,35)}")
        self.m_btnEnter2.setStyleSheet("QPushButton{color:white;background:rgb(35,35,35)}")
        self.m_btnEnter3.setStyleSheet("QPushButton{color:white;background:rgb(35,35,35)}")
        self.m_btnEnter4.setStyleSheet("QPushButton{color:white;background:rgb(35,35,35)}")
        self.lineEdit.setStyleSheet("color:white")
        self.plainTextEdit.setStyleSheet("color:white")
        self.label2.setStyleSheet("color:white")
        self.label3.setStyleSheet("color:white")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        #MainWindow.resize(1500, 850)

        MainWindow.setStyleSheet("background-color:rgb(75,75,75);")



        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))



        #self.m_btnEnter = QtGui.QPushButton(self.centralwidget)
        #self.m_btnEnter.setGeometry(QtCore.QRect(40, 90, 99, 27))
        #self.m_btnEnter.setObjectName(_fromUtf8("m_btnEnter"))
        self.m_btnEnter2 = QtGui.QPushButton(self.centralwidget)
        #self.m_btnEnter2.setGeometry(QtCore.QRect(40, 140, 99, 27))
        self.m_btnEnter2.setObjectName(_fromUtf8("m_btnEnter2"))

        self.m_btnEnter3 = QtGui.QPushButton(self.centralwidget)
        # self.m_btnEnter2.setGeometry(QtCore.QRect(40, 140, 99, 27))
        self.m_btnEnter3.setObjectName(_fromUtf8("m_btnEnter3"))


        self.m_btnEnter4 = QtGui.QPushButton(self.centralwidget)
        # self.m_btnEnter2.setGeometry(QtCore.QRect(40, 140, 99, 27))
        self.m_btnEnter4.setObjectName(_fromUtf8("m_btnEnter4"))

        self.label2 = QtGui.QLabel(self.centralwidget)
        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setObjectName(_fromUtf8("label2"))


        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))



        self.label3 = QtGui.QLabel(self.centralwidget)
        self.label3.setAlignment(QtCore.Qt.AlignCenter)
        self.label3.setObjectName(_fromUtf8("label3"))

        self.plainTextEdit = QtGui.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setObjectName(_fromUtf8("plainTextEdit"))
        self.plainTextEdit.setMinimumHeight(480)


        self.label = QtGui.QLabel(self.centralwidget)
        #self.label.setGeometry(QtCore.QRect(180, 90, 1280, 720))
        self.label.setText(_fromUtf8(""))
        self.label.setObjectName(_fromUtf8("label"))
        self.label.setMinimumSize(1280,720)
        #MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        #self.menubar.setGeometry(QtCore.QRect(0, 0, 1500, 31))


        self.verticalLayout = QtGui.QVBoxLayout()
        #self.verticalLayout.setMargin(11)
        self.verticalLayout.setSpacing(8)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))

        self.verticalLayout.addWidget(self.label2)
        self.verticalLayout.addWidget(self.lineEdit)

        self.verticalLayout.addWidget(self.m_btnEnter3)
        #self.verticalLayout.addWidget(self.m_btnEnter)
        self.verticalLayout.addWidget(self.m_btnEnter2)

        self.verticalLayout.addWidget(self.m_btnEnter4)

        self.verticalLayout.addWidget(self.label3)
        self.verticalLayout.addWidget(self.plainTextEdit)




        #定义弹簧和加进verticalLayout
        spacerItem = QtGui.QSpacerItem(120, 50, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)


        #定义gridLayout，把verticalLayout和QLabel加进去，整个设为gridLayout
        self.gridlayout = QtGui.QGridLayout()
        self.gridlayout.setSpacing(15)
        self.gridlayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.label,0,1,1,1)

        MainWindow.setLayout(self.gridlayout)

        #self.menubar.setObjectName(_fromUtf8("menubar"))
        #MainWindow.setMenuBar(self.menubar)
        #self.statusbar = QtGui.QStatusBar(MainWindow)
        #self.statusbar.setObjectName(_fromUtf8("statusbar"))
        #MainWindow.setStatusBar(self.statusbar)

        self.setUpUiStyle()

        self.retranslateUi(MainWindow)
        #QtCore.QObject.connect(self.m_btnEnter, QtCore.SIGNAL(_fromUtf8("clicked()")), self.showVideo1)
        QtCore.QObject.connect(self.m_btnEnter2, QtCore.SIGNAL(_fromUtf8("clicked()")), self.change_videoflag)
        QtCore.QObject.connect(self.m_btnEnter3, QtCore.SIGNAL(_fromUtf8("clicked()")), self.addface)
        QtCore.QObject.connect(self.m_btnEnter4, QtCore.SIGNAL(_fromUtf8("clicked()")), self.findpeopleface)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        #self.m_btnEnter.setText(_translate("MainWindow", "播放视频", None))
        self.m_btnEnter2.setText(_translate("MainWindow", "暂停播放", None))
        self.m_btnEnter3.setText(_translate("MainWindow", "添加人脸", None))
        self.m_btnEnter4.setText(_translate("MainWindow", "识别人脸", None))
        self.label2.setText(_translate("MainWindow", "录入名称", None))
        self.label3.setText(_translate("MainWindow", "检测结果", None))


if __name__=="__main__":
    import sys
    app=QtGui.QApplication(sys.argv)
    widget=QtGui.QWidget()
    ui=Ui_MainWindow()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())