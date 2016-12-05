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
		
	'''	
	This function serves as a common resource for all flow feature vector functions. It calculates the optical flow for the video given a maximum
	number of features to track. It then calculates the distance between the initial location of the feature (corner) and the ending location. 
	This distance is considered the magnitude of the optical flow. The angle (direction) is also calculated
	'''
	def commomFlowFeatureVector(self, maxFeat):
		# params for ShiTomasi corner detection	
		feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

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
	
	
	'''
	meanFlowFeatureVector finds the mean of the distances and the mean of the angles and returns a vector of size two with these two features
	to represent the entire flow.
	'''		
	def meanFlowFeatureVector(self, maxFeat):
		distanceFeatures, angleFeatures = self.commomFlowFeatureVector(maxFeat)
		
		meanDistance = sum(distanceFeatures)/(1.0*len(distanceFeatures))
		meanAngles = sum(angleFeatures)/(1.0*len(angleFeatures))
		
		meanFlowFeatureVector = [meanDistance, meanAngles]
		
		return meanFlowFeatureVector
	
	'''
	allFlowFeatureVector returns all the distances and angles.
	note that since the goodfeaturestoTrack function does not always return the same number of features (corners), this function fills 
	each vector with zeros so that they all have the same size of 2*maxFeat. |distances| = maxFeat and |angles| = maxFeat
	'''
	
	def allFlowFeatureVector(self, maxFeat):
		distanceFeatures, angleFeatures = self.commomFlowFeatureVector(maxFeat)
		
		#filling the array so that all of them have the same length (maxFeat)
		l = maxFeat -len(distanceFeatures)
		distanceFeatures = distanceFeatures + [0.00000000e+00]*l
		angleFeatures = angleFeatures + [0.00000000e+00]*l
			
		allFlowFeatureVector = distanceFeatures + angleFeatures
		return allFlowFeatureVector
	
	
	'''
	minMaxFlowFeatureVector calculates the average distance, average angle, minimum distance, minimum angle, maximum distance, and maximum angle
	It returns a vector of size 6 to describe the overall flow.
	'''
	def minMaxFlowFeatureVector(self, maxFeat):
		distanceFeatures, angleFeatures = self.commomFlowFeatureVector(maxFeat)
		
		meanDistance = sum(distanceFeatures)/(1.0*len(distanceFeatures))
		meanAngles = sum(angleFeatures)/(1.0*len(angleFeatures))
		
		maxDistance = max(distanceFeatures)
		minDistance = min(distanceFeatures)
		
		maxAngle = max(angleFeatures)
		minAngle = min(angleFeatures)

		
		minMaxFlowFeatureVector = [meanDistance, meanAngles, maxDistance, minDistance, maxAngle, minAngle]
		return minMaxFlowFeatureVector
	
		
		
