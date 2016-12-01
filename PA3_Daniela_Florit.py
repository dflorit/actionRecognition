from util import Util
from featureExtractors import *
from classifiers import *
from numpy import *
import cv2

class Assignment3:
	
	def __init__(self):
		self.u = Util()
		self.svn = SVMClassifier()
		self.path = "ucf_sports_actions/ucf action"
		
		#list of all directory paths to get the videos
		#list of the labels (number associated to each category)
		self.videoPaths, self.videoLabels = self.u.getVideoPaths(self.path)
		self.meanFlowFeatureVector = []
		
		for video in self.videoPaths:
			v = FeatureExtractors(video)
			features = v.meanFlowFeatureVector()
			self.meanFlowFeatureVector.append(features)
	
		#shuffle the feature vector and the labels:
		l = len(self.videoLabels)
		index = [x for x in range(l)]
		random.shuffle(index)
		self.videoLabels = [self.videoLabels[x] for x in index]
		self.meanFlowFeatureVector = [self.meanFlowFeatureVector[x] for x in index]
		
		
		#self.printStuff()
		
	
	def printStuff(self):
		print "meanFlowFeatureVector:"
		print self.meanFlowFeatureVector
		print "\n"
		print "video labels:"
		print self.videoLabels
		print "\n"
		print len(self.videoLabels), len(self.meanFlowFeatureVector), len(self.videoPaths)
		
	
	
		
		
		
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PA3 = Assignment3()


