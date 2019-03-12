import numpy as np
import cv2

class CamshiftTracker(object):
    def __init__(self,roiPts):
        #set initial points
        self.roiPts = np.array(roiPts)
        self.hist = None
        self.roiBox = None

        #setup termination criteria, either 10 iterations or move by atlease 1 pt
        self.termCrit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

    def orderPoints(self):
        assert len(self.roiPts)==2
        return self.roiPts[np.argsort(self.roiPts.sum(axis=1))]

    def track(self,image):
        self.image = image
        hsv = cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)

        #mask for better tracking
        mask = cv2.inRange(hsv,np.array((0.,60.,32.)),np.array((180.,255.,255.)))

        #order the points and gather top-left and bottom-right points
        pts = self.orderPoints().tolist()
        tl,br = pts

        #if tracking is not yet started initialize it by setting the roiBox and hist
        if self.roiBox is None:
            self.roiBox = (tl[0],tl[1],br[0]-tl[0],br[1]-tl[1])

        if self.hist is None:
            hsv_roi = hsv[tl[1]:br[1],tl[0]:br[0]]
            mask_roi = mask[tl[1]:br[1],tl[0]:br[0]]
            self.hist = cv2.calcHist([hsv_roi],[0],mask_roi,[16],[0,180])
            self.hist = cv2.normalize(self.hist,self.hist,0,255,cv2.NORM_MINMAX)
            self.hist = self.hist.reshape(-1)

        try:
            #backproject the histogram on to the original histogram
            prob = cv2.calcBackProject([hsv],[0],self.hist,[0,180],1)
            prob &= mask

            #get location of the object in the new frame
            trackBox,self.roiBox = cv2.CamShift(prob, self.roiBox, self.termCrit)
            return self.roiBox

        except Exception:
            return None