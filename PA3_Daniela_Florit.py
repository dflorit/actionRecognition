from util import Util
from featureExtractors import *
from classifiers import *
from numpy import *
import cv2
import sys

class Assignment3:
	
	def __init__(self):
		self.u = Util()
		
		#If the videos are in a different path, please change this line accordingly 
		self.path = "ucf_sports_actions/ucf action"
		
		#labels used to represent each category. Note that throughout the program, only the number labels are used
		self.category_labels = {0:'Diving', 1:'Golf-Swing', 2:'Kicking', 3:'Lifting', 4:'Riding-Horse', 5:'Running', \
		6:'SkateBoarding', 7:'Swing-Bench', 8:'Swing-Side', 9:'Walking'}
		
		#list of all directory paths to get the videos and list of the labels (number associated to each category)
		self.videoPaths, self.videoLabels = self.u.getVideoPaths(self.path)
		self.numVideos = len(self.videoPaths)
	
	
	'''
	This function calculates the feature vectors for ALL videos, and puts them in a list. 
	The index matches the index of the videos in videoLabels.
	Note that it receives a maxFeat number, which indicates the maximum number of features to track. This can be changed on each experiment.
	It finds the features according to three different modes: 
		"all" includes the distances and angles from a maxFeat number features tracked throughout the video.
		"mean" includes the average distance and average angle from the 100 features
		"minMax" includes the average distance, average angle, minimum distance, minimum angle, maximum distance, and maximum angle
		if the mode is none of these three, an Exception will rise
	'''
	def getFeatureVector(self, maxFeat, mode = "all"):

		self.featureVector = []
		for i, video in enumerate(self.videoPaths):
			print "Extracting features from video ", i
			sys.stdout.write("\033[F") # Cursor up one line
			
			v = FeatureExtractors(video)
			if mode == "all":
				features = v.allFlowFeatureVector(maxFeat) 
			elif mode == "minMax":
				features = v.minMaxFlowFeatureVector(maxFeat)
			elif mode == "mean":
				features = v.meanFlowFeatureVector(maxFeat)
			else:
				raise Exception("This mode is not supported")

			self.featureVector.append(features)
		
		self.featureVector = array(self.featureVector)
		print "\n"
	
	
	'''
	This function gets the Metrics for each experiment after it runs. It uses the confussion matrix found during the cross-evaluation and 
	calculates the True Positives(TP), True Negatives(TN), False Positives(FP), False Negatives(FN) for each class. These are arrays of 
	length 10, each index represents the corresponding class (0-9). See self.category_labels for the mapping. 
	It calculates the overall accuracy of the experiment, and the precision, sensitivity, and specificity for each action class. 
	All but accuracy are also arrays in which the indices map to each action class. 
	'''
	def getMetrics(self):
		
		accuracy = 0
		TP = zeros(10)
		TN = zeros(10)
		FP = zeros(10)
		FN = zeros(10)
		precision = zeros(10)
		sensitivity = zeros(10)
		specificity = zeros(10)
		
		for i in range(10): #possible classes Ground Truth 
			TP[i] = self.confusionMatrix[(i,i)]
			TN[i] = sum([self.confusionMatrix[x,y] for x in range(10) for y in range(10) if x != i if y != i])
			FP[i] = sum([self.confusionMatrix[x,y] for x in range(10) for y in range(10) if x != i if y == i])
			FN[i] = sum([self.confusionMatrix[x,y] for x in range(10) for y in range(10) if x == i if y != i])
			
			#to avoid dividing by zero:
			if TP[i] == 0:
				TP[i] = 0.001
			if TN[i] == 0:
				TN[i] = 0.001
			if FP[i] == 0:
				FP[i] = 0.001
			if FN[i] == 0:
				FN[i] = 0.001
			
			#calculating precision, specificity and sensitivity for each class:
			precision[i] = 1.0*TP[i] / (1.0*(TP[i] + FP[i]))
			sensitivity[i] = 1.0*TP[i] / (1.0*(TP[i] + FN[i]))
			specificity[i] = 1.0*TN[i] / (1.0*(TN[i] + FP[i]))
		
		#overall accuracy:
		accuracy = 1.0 * (sum(TP) + sum(TP)) / (sum(TP) + sum(TP) + sum(FP) + sum(FN))
		
		
		#printing metrics for each category:
		print "\n"
		print "Printing Metrics"
		print "The accuracy for this test is: ", accuracy, "\n"
		
		for key in self.category_labels:
			print "Metrics for : ", self.category_labels[key]
			print "precision: ", precision[key]
			print "sensitivity: ", sensitivity[key]
			print "specificity: ", specificity[key]
			print "\n"
	
	
	'''
	This is the main function which performs LOO cross-validation. 
	This makes it more general and can be used for different experiments.
	The parameters nnAlpha and nnHiddenLayersNumber are used only for neural networks
	The parameters SVMkernel, and SVMdegree are used only when the classifier is SVM. SVMdegree is used only when SVMkernel is set to 'poly'
	
	In LOO cross-validation, the program leaves one video out to be used as the test sample, and trains with the rest. This is repeated
	self.numVideos times (number of videos), each time leaving out a different video and training with the rest. 
	
	A confusion matrix is built during the cross-validation. It's a dictionary with all combinations of (ground Truth, prediction) as keys.
	This is later used for metrics. 
	
	It initializes the classifier based on the classifierMode parameter:
		'SVM' uses svm.SVC() classifier which assumes a 'one vs one' approach. It sets the kernel to SVMkernel, and the degree to SVMdegree
		'LinearSVM' uses svm.LinearSVC() classifier which assumes a 'one vs rest' approach. Default parameters are maintained
		'NN' uses MLPClassifier() classifier which uses a neural network with alpha = nnAlpha, and nnHiddenLayersNumber hidden layers
		if the mode is none of these three, an Exception will rise
	
	'''			
	def evaluate(self, maxFeat, FeatureVectorMode, classifierMode, nnAlpha = None, nnHiddenLayersNumber = None, \
	SVMkernel = None, SVMdegree = None):
		
		self.getFeatureVector(maxFeat, FeatureVectorMode)
		self.confusionMatrix = {}
		self.accuracy = 0
		self.GT = 0
		
		#initializing confusion matrix
		for i in range(10):  #possible classes Ground Truth 
			for j in range(10): #possible classes predicted
				self.confusionMatrix[(i,j)] = 0
		
		#LOO cross-validation:
		for i in range(self.numVideos):
			#video being left out for testing is at index i
			videoFeaturesTesting = self.featureVector[i]
			videoLabelsTestingGT = self.videoLabels[i]
			
			#all videos not at index i are being included for training:
			videoFeaturesTraining = [self.featureVector[x] for x in range(self.numVideos) if x != i]
			videoLabelsTraining = [self.videoLabels[x] for x in range(self.numVideos) if x != i]
			
			#shuffle the feature vector and the labels:
			l = len(videoFeaturesTraining)
			index = [x for x in range(l)]
			random.shuffle(index)
			videoLabelsTraining = [videoLabelsTraining[x] for x in index]
			videoFeaturesTraining = [videoFeaturesTraining[x] for x in index]
			
			#initialize classifier according to classifierMode:
			if classifierMode == "SVM":
				clf = SVCClassifier(SVMkernel, SVMdegree)
			elif classifierMode == "LinearSVM":
				clf = SVCLinearClassifier()
			elif classifierMode == "NN":
				clf = NNClassifier(nnAlpha, nnHiddenLayersNumber)
			else:
				raise Exception("This classifier mode is not supported")
			
			#training step using SVM:
			print "Training and testing iteration ", i+1
			sys.stdout.write("\033[F") # Cursor up one line

						
			clf.training(videoFeaturesTraining, videoLabelsTraining)
		
			#predicting:
			predictedLabel = clf.test(videoFeaturesTesting)
			self.confusionMatrix[(videoLabelsTestingGT, predictedLabel[0])] += 1
		
		#getting classifier parameters:
		if classifierMode != 'NN':
			params = clf.getParameters()
			print "\n Parameters for classifier:"
			print params
		
		#Calculating metrics for this test:
		self.getMetrics()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Experiments:
	def __init__(self):	
		self.FeatureVectorMode = ["all", "mean", "minMax"]
		self.classifierMode = ["SVM", "LinearSVM", "NN"]
		self.SVMkernel = ['linear', 'sigmoid', 'poly', 'rbf']
		
		#defaults:
		self.maxFeat = 100
		self.nnAlpha = 1e-5
		self.nnHiddenLayersNumber = 15
		self.SVMdegree = 3
	
	def printDescription(self, experimentNum, maxFeat, FeatureVectorMode, classifierMode, nnAlpha, nnH, SVMkernel, SVMdegree, description):
		print "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
		print "Experiment ", experimentNum
		print description
		print "\n"
		
		print "Feature vector used: ", FeatureVectorMode
		print "Classifier used: ", classifierMode
		print "Max number of features (corners) extracted from video: ", maxFeat
		
		
		if classifierMode == 'SVM':
			print "Relevant parameters: "
			print "	kernel used: ", SVMkernel
			if SVMkernel == 'poly':
				print "	degree used: ", SVMdegree
		elif classifierMode == 'NN':
			print "Relevant parameters: "
			print '	alpha used: ', nnAlpha
			print '	number of hidden layers ', nnH
		print "\n"
	
	
	#experiment 1 uses SVM classifier with 'all' feature vector and all default parameters	
	def experiment1(self):
		ex1 = Assignment3()
		description = "Experiment 1 uses 'SVM' classifier with 'all' feature vector and all default parameters"
		
		self.printDescription(1, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex1.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 2 uses SVM classifier with 'mean' feature vector and all default parameters
	def experiment2(self):
		ex2 = Assignment3()
		description = "Experiment 2 uses 'SVM' classifier with 'mean' feature vector and all default parameters"
		
		self.printDescription(2, self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex2.evaluate(self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 3 uses SVM classifier with 'minMax' feature vector and all default parameters		
	def experiment3(self):
		ex3 = Assignment3()
		description = "Experiment 3 uses 'SVM' classifier with 'minMax' feature vector and all default parameters"
		
		self.printDescription(3, self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex3.evaluate(self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 4 uses LinearSVM classifier with 'all' feature vector and all default parameters		
	def experiment4(self):
		ex4 = Assignment3()
		description = "Experiment 4 uses 'LinearSVM' classifier with 'all' feature vector and all default parameters"
		
		self.printDescription(4, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex4.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 5 uses LinearSVM classifier with 'mean' feature vector and all default parameters		
	def experiment5(self):
		ex5 = Assignment3()
		description = "Experiment 5 uses 'LinearSVM' classifier with 'mean' feature vector and all default parameters"
		
		self.printDescription(5, self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex5.evaluate(self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 6 uses LinearSVM classifier with 'minMax' feature vector and all default parameters		
	def experiment6(self):
		ex6 = Assignment3()
		description = "Experiment 6 uses 'LinearSVM' classifier with 'minMax' feature vector and all default parameters"
		
		self.printDescription(6, self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex6.evaluate(self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[1], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 7 uses 'NN' classifier with 'all' feature vector and all default parameters		
	def experiment7(self):
		ex7 = Assignment3()
		description = "Experiment 7 uses 'NN' classifier with 'all' feature vector and all default parameters"
		
		self.printDescription(7, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex7.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 8 uses 'NN' classifier with 'mean' feature vector and all default parameters		
	def experiment8(self):
		ex8 = Assignment3()
		description = "Experiment 8 uses 'NN' classifier with 'mean' feature vector and all default parameters"
		
		self.printDescription(8, self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex8.evaluate(self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#experiment 9 uses 'NN' classifier with 'minMax' feature vector and all default parameters		
	def experiment9(self):
		ex9 = Assignment3()
		description = "Experiment 9 uses 'NN' classifier with 'minMax' feature vector and all default parameters"
		
		self.printDescription(9, self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex9.evaluate(self.maxFeat, self.FeatureVectorMode[2], self.classifierMode[2], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
		
	
	#experiment 10 uses SVM classifier with 'all' feature vector with 'linear' kernell and all other default parameters	
	def experiment10(self):
		ex10 = Assignment3()
		description = "Experiment 10 uses SVM classifier with 'all' feature vector with 'linear' kernell and all other default parameters"
		
		self.printDescription(10, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[0], self.SVMdegree, description)
		
		ex10.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[0], self.SVMdegree)
	
	
	#experiment 11 uses SVM classifier with 'all' feature vector with 'poly' kernell of defaul degree 3 and all other default parameters	
	def experiment11(self):
		ex11 = Assignment3()
		description = "Experiment 11 uses SVM classifier with 'all' feature vector with 'poly' kernell of defaul degree 3 \
		and all other default parameters"
		
		self.printDescription(11, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[2], self.SVMdegree, description)
		
		ex11.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[2], self.SVMdegree)
	
	#experiment 12 uses 'SVM' classifier with 'all' feature vector with 'poly' kernell of degree 6 and all other default parameters	
	def experiment12(self):
		ex12 = Assignment3()
		description = "Experiment 12 uses 'SVM' classifier with 'all' feature vector with 'poly' kernell of degree 6 \
		and all other default parameters"
		SVMdegree = 6
		
		self.printDescription(12, self.maxFeat, self.FeatureVectorMode[1], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[2], SVMdegree, description)
		
		ex12.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[0], self.nnAlpha, self.nnHiddenLayersNumber,\
		self.SVMkernel[2], SVMdegree)
	
	#Experiment 13 uses 'NN' classifier with 'all' feature vector with 50 hidden layers and all other default parameters		
	def experiment13(self):
		ex13 = Assignment3()
		description = "Experiment 13 uses 'NN' classifier with 'all' feature vector with 50 hidden layers and all other default parameters"
		nnHiddenLayersNumber = 50
		
		self.printDescription(13, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex13.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#Experiment 14 uses 'NN' classifier with 'all' feature vector with 150 hidden layers and all other default parameters		
	def experiment14(self):
		ex14 = Assignment3()
		description = "Experiment 14 uses 'NN' classifier with 'all' feature vector with 150 hidden layers and all other default parameters"
		nnHiddenLayersNumber = 150
		
		self.printDescription(14, self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree, description)
		
		ex14.evaluate(self.maxFeat, self.FeatureVectorMode[0], self.classifierMode[2], self.nnAlpha, nnHiddenLayersNumber,\
		self.SVMkernel[3], self.SVMdegree)
	
	#all experiments:
	def allExperiments(self):
		#experiments 1-9 are all combinations of the three classifiers and three feature vectors, with default parameters
		#self.experiment1()
		#self.experiment2()
		#self.experiment3()
		#self.experiment4()
		#self.experiment5()
		#self.experiment6()
		#self.experiment7()
		#self.experiment8()
		#self.experiment9()
		
		#experimenting with SVM classifier with different kernels:
		#self.experiment10()
		#self.experiment11()
		
		#This experiment takes too long. I recommend not using it:
		#self.experiment12()
		
		#experimenting with NN classifier with different number of hidden layers:
		#self.experiment13()
		self.experiment14()
		
		
		

PA3 = Experiments()
PA3.allExperiments()
	


