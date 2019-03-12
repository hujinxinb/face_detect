import dlib
import numpy as np

class CorrelationTracker(object):

    def __init__(self,roiPts):
        #set initial points
        self.roiPts = np.array(roiPts)
        self.tracker = None

    def orderPoints(self):
        assert len(self.roiPts)==2
        return self.roiPts[np.argsort(self.roiPts.sum(axis=1))]

    def track(self,image):
        #create a new tracker
        if self.tracker is None:
            self.tracker = dlib.correlation_tracker()
            pts = self.orderPoints().tolist()
            tl, br = pts
            (x,y,w,h) = (tl[0],tl[1],br[0]-tl[0],br[1]-tl[1])
            roi_pts = [x,y,x+w,y+h]
            self.tracker.start_track(image,dlib.rectangle(*roi_pts))

        #update the tracker with current frame and get the current estimated position
        self.tracker.update(image)
        pts = self.tracker.get_position()
        (x,y,xb,yb) = (pts.left(),pts.top(),pts.right(),pts.bottom())

        #return the points of the form (x,y,w,h)
        return np.int0((x,y,(xb-x),(yb-y)))