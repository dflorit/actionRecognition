from numpy import *
from glob import *
from os import *
import cv2


class Util:
	
	#returns the pythagorean distance between two coordinates:
	def calculateDistance(self, xy1, xy2):
		(x1, y1) = xy1
		(x2, y2) = xy2
		dist = sqrt((x2-x1)**2 + (y2-y1)**2)
		return dist
	
	#returns an angle in radians from 0 to 2pi (no negative values)
	def calculateAngle(self, p1, p2):
		x1, y1 = p1
		x2, y2 = p2
		angle = arctan2(y2-y1, x2-x1) 
		if angle < 0:
			angle = 2*pi + angle
		return angle	
	
	#function to get the videos
	#note that I am assuming that the paths to get the videos are exactly those found on the folder that is downloaded form the dataset website.
	def getVideoPaths(self, path):
		
		#mapping each of the 10 category names to a number to be used as the label (0 to 9)
		category_labels = {0:'Diving', 1:'Golf-Swing', 2:'Kicking', 3:'Lifting', 4:'Riding-Horse', 5:'Running', 6:'SkateBoarding', \
		 7:'Swing-Bench', 8:'Swing-Side', 9:'Walking'}
		
		#list of category name for each folder
		category_folders = {'Diving-Side':0, 'Golf-Swing-Back':1, 'Golf-Swing-Front':1, 'Golf-Swing-Side':1, 'Kicking-Front':2, \
		'Kicking-Side':2, 'Lifting':3, 'Riding-Horse':4, 'Run-Side':5, 'SkateBoarding-Front':6, 'Swing-Bench':7, 'Swing-SideAngle':8, \
		'Walk-Front':9}
		
		videoPaths = []
		videoLabels = []
		categoryFolders = listdir(path)
		
		for cat in categoryFolders:
			if cat != ".DS_Store" :
				category = path + "/" + cat
				videoFolders = listdir(category)
				for video in videoFolders:
					folder = category + "/" + video
					f = glob(folder + '/*.avi')
				
					if f != []:
						videoPaths += f
						videoLabels.append(category_folders[str(cat)])
		return videoPaths, videoLabels
	
