from util import Util
from numpy import *
from math import *
import cv2

'''
Please note that some pieces of this code have been copied from the examples provided in the OpenCV documentation for optical flow. The necessary modifications have been done to adapt it to this problem.
However, some of the parameters, variable names, and functions have been kept.
'''

class FeatureExtractors:
	
	def __init__(self, videopath):
		self.u = Util()
		self.path = videopath
		self.cap = cv2.VideoCapture(self.path)
	
	def commomFlowFeatureVector(self):
		# params for ShiTomasi corner detection	
		feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

		# Parameters for lucas kanade optical flow
		lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
		# Take first frame and find corners in it
		ret, old_frame = self.cap.read()
		old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		
		#features to track
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
		initialFeatures = p0.copy()
		
		#finding optical flow between every pair of consecutive frames
		while(1):
			ret, frame = self.cap.read()
			
			if ret == False:
				break
			
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			#calculate optical flow:
			p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
			
			#update the previous frame and points:
			old_gray = frame_gray.copy()
			p0 = p1.reshape(-1,1,2)
		
		
		#creating the feature vector:
		
		distanceFeatures = []
		angleFeatures = []
		
		for i in range(len(initialFeatures)):
			x1,y1 = initialFeatures[i][0].ravel()
			x2,y2 = p0[i][0].ravel()
			
			distance = self.u.calculateDistance((x1,y1), (x2,y2))
			distanceFeatures.append(distance)
			
			angle = self.u.calculateAngle((x1,y1), (x2,y2))
			angleFeatures.append(angle)
		
		cv2.destroyAllWindows()
		self.cap.release()
		return distanceFeatures, angleFeatures
			
	def meanFlowFeatureVector(self):
		distanceFeatures, angleFeatures = self.commomFlowFeatureVector()
		meanDistance = sum(distanceFeatures)/(1.0*len(distanceFeatures))
		meanAngles = sum(angleFeatures)/(1.0*len(angleFeatures))
		meanFlowFeatureVector = [meanDistance, meanAngles]
		
		return meanFlowFeatureVector
	
	def allFlowFeatureVector(self):
		distanceFeatures, angleFeatures = self.commomFlowFeatureVector()
		allFlowFeatureVector = distanceFeatures + angleFeatures
		
		print len(allFlowFeatureVector)
		
		return allFlowFeatureVector
		
