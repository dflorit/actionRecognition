from PIL import Image
from numpy import *
from scipy.signal import *
from scipy.misc import *
import cv2
from skimage.transform import pyramid_gaussian, pyramid_expand, warp

class Util:
	
	def __init__(self):
		self.xDerMask = array([[-1,1],[-1,1]])
		self.yDerMask = array([[-1,-1],[1,1]])
		self.tDerMask1 = array([[-1,-1],[-1,-1]])
		self.tDerMask2 = array([[1,1],[1,1]])
	
	#utility functions for reading and writing images to file
	def readImg(self, dir): #returns an array
		return imread(dir, 1)

	def readImgRGBArray(self, dir): #returns an array
		im = Image.open(dir)
		rgb_im = im.convert('RGB')
		rgb_arr =array(rgb_im)
		return rgb_arr
	
	def readImgRGB(self, dir): #returns an array
		im = Image.open(dir)
		rgb_im = im.convert('RGB')
		return rgb_im
		
	def saveImg(self, dir, img):
		imsave(dir, img)		
		
	def saveImgRGB(self, dir, img): #getting img as an array, need to convert to picture
		img2 = Image.fromarray(img)
		img2.save(dir)
	# ----------------------------------------------------
			
	#do convolution between image I and mask h: 
	def convolve(self, I, h):
		return convolve2d(I, h, 'same')
	
	#save results for canny
	def saveImageResults(self, xs):
		for img, dir in xs:
			self.saveImg(dir, img)
	
	#Computing the magnitudes of Ix and Iy:
	def getMagnitude(self, Ix, Iy):
		x = len(Ix)
		y = len(Ix[0])
		magnitudes = zeros([x,y])
		
		for i in range(x):
			for j in range(y):
				magnitudes[i][j] = sqrt(Ix[i][j]**2 + Iy[i][j]**2) 
				
		return magnitudes
	
	#formula to get values of gaussian:
	def gaussianValues(self, xs, sigma, d = 0): #former gaussFunction1d
	
		if d == 0: #want just the gaussian
			return exp(-1.0*(xs**2/(2.0*sigma**2)))
		elif d == 1: #want the first derivative
			return (-1.0*xs/(sigma**2))*exp(-1.0*(xs**2/(2.0*sigma**2)))

	#formula to get values of gaussian for 2d:
	def gaussianValues2d(self, x, y, sigma, mode  = 'G'):
		
		g = (exp(-1.0*((x**2+y**2)/(2*sigma**2))))*(1.0/(sqrt(2.0*pi)*sigma))
		
		if mode == 'G':
			return g
		elif mode == 'Gx': 
			Gx = g * (-1.0*x/sigma**2)
			return Gx
			
		elif mode == 'Gy': 
			Gy = g * (-1.0*y/sigma**2)
			return Gy
			
		elif mode == 'Gxx':
			Gxx = g * (1.0/sigma**2) * (1.0*(x**2)/(sigma**2)-1)
			return Gxx
			
		elif mode == 'Gxy':
			Gxy = g*(1.0*x*y/sigma**4)
			return Gxy
			
		elif mode == 'Gyy':   
			Gyy = g * (1.0/sigma**2) * (1.0*(y**2)/(sigma**2)-1)
			return Gyy
			
				
	#Creating Gaussian Mask in x and y directions:
	def getGaussian(self, sigma, d = 0):
		#these are the x values that will be used for the gaussian 
		l_2 = 3*int(ceil(sigma))
		xs = [x for x in range(-l_2, l_2+1)]
		
		gaussian_x = array([self.gaussianValues(array(xs), sigma, d)])
		gaussian_y = gaussian_x.transpose()
		return gaussian_x, gaussian_y
	
	def getGaussian2d(self, sigma, mode = 'G'):
		l = 6*int(ceil(sigma))+1
		l_2 = 3*int(ceil(sigma))
		gaussian2d = zeros([l,l])
		xs = [[(x,y) for x in range(-l_2,l_2+1)] for y in range(-l_2,l_2+1)]
		
		for i, row in enumerate(xs):
			for j, column in enumerate(row):
				gaussian2d[i][j] = self.gaussianValues2d(column[0], column[1], sigma, mode)
		
		return gaussian2d
	
	
	
	def calculateDistance(self, xy1, xy2):
		(x1, y1) = xy1
		(x2, y2) = xy2
		dist = sqrt((x2-x1)**2 + (y2-y1)**2)
		return dist
	
	#getting all neighbors for a given pixel in position x,y	
	def right_neighbor(self, x, y, I):
		return (I[x][y+1], (x, y+1))
	
	def top_neighbor(self, x, y, I):
		return (I[x-1][y], (x-1, y))
	
	def left_neighbor(self, x, y, I):
		return (I[x-1][y], (x-1, y))
	
	def bottom_neighbor(self, x, y, I):
		return (I[x+1][y], (x+1, y))
	
	def left_neighbor(self, x, y, I):
		return (I[x][y-1], (x, y-1))
		
	def rightTop_neighbor(self, x, y, I):
		return (I[x-1][y+1], (x-1, y+1))
	
	def rightBottom_neighbor(self, x, y, I):
		return (I[x+1][y+1], (x+1, y+1))
	
	def leftTop_neighbor(self, x, y, I):
		return (I[x-1][y-1], (x-1, y-1))
	
	def leftBottom_neighbor(self, x, y, I):
		return (I[x+1][y-1], (x+1, y-1))
	
	def getAllNeighbors(self, position, I):
		(x, y) = position
		neighbors = []
		
		if x != 0: #it's not the first line
			neighbors.append(self.top_neighbor(x, y, I))
		if x != len(I) -1: #it's not the last line
			neighbors.append(self.bottom_neighbor(x, y, I))
		if y != 0: #it's not the first column
			neighbors.append(self.left_neighbor(x, y, I))
			
			if x != 0: 
				neighbors.append(self.leftTop_neighbor(x, y, I))
			if x != len(I) -1:
				neighbors.append(self.leftBottom_neighbor(x, y, I))
		if y != len(I[0])-1:
			neighbors.append(self.right_neighbor(x, y, I))
			if x != 0: 
				neighbors.append(self.rightTop_neighbor(x, y, I))
			if x != len(I) -1:
				neighbors.append(self.rightBottom_neighbor(x, y, I))
		return neighbors
	# ----------------------------------------------------
	
	def sumation(self, M, positions):
		sum = 0.0
		#print "positions: ", positions
		for pos in positions:
			#print "pos ", pos
			sum = sum + M[pos]
		#print sum
		return sum
	
	def computeFx(self, I1, I2):
		return self.convolve(I1, self.xDerMask) + self.convolve(I2, self.xDerMask)
	
	def computeFy(self, I1, I2):
		return self.convolve(I1, self.yDerMask) + self.convolve(I2, self.yDerMask)
	
	def computeFt(self, I1, I2):
		return self.convolve(I1, self.tDerMask1) + self.convolve(I2, self.tDerMask2)
	
	#get 3x3 matrix of neighbors surrounding a position
	def get3x3Matrix(self, position, img):
		M3x3 = [position]
		neighs = self.getAllNeighbors(position, img)
		for n in neighs:
			M3x3.append(n[1])
		return M3x3
	
	#generate random RGB colors
	def getRandomColorList(self, length):
		colors = []
		
		for i in range(length):
			c1 = random.randint(0,255)
			c2 = random.randint(0,255)
			c3 = random.randint(0,255)
			colors.append((c1,c2,c3))
		
		return colors
	
	#I am using this function to find the shape of the dictionary, to be used later when 
	#converting it to an array
	def findShapeDict(self, dict):
		
		maxi = 0
		maxj = 0
		for key in dict:
			if key[0] > maxi:
				maxi = key[0]
			if key[1] > maxj:
				maxj = key[1]
		
		return maxi+1, maxj+1
	
	#since my LukasKanade function returns a dictionary, and I need to use it as an array, 
	#I am using this function to convert from dictionary to array
	def arrayFromDict(self, dict):
		lx, ly = self.findShapeDict(dict)
		arr = []
		for i in range(lx):
			row = []
			for j in range(ly):
				col = dict[(i,j)]
				row.append(col)
			arr.append(row)
		return array(arr)
	
	
	#this function creates a translation function for warp, and calls warp.
	#By doing this, I am warping the Image at pyramid level k with the expanded flow
	#obtained from the k-1 level: 
	def warpFunction(self, I1, u, v):
		
		def flow_translation(xy):
			xyn = []
			for coords in xy:
				newX = coords[0] - v[coords[1]][coords[0]]
				newY = coords[1] - u[coords[1]][coords[0]]
				xyn.append([newX, newY])
			xyn = array(xyn)
			return xyn
		
		I_warped = warp(I1, flow_translation)
		return I_warped
	
	
	#calculates the norm at a point. ||x||
	def getNorm(self, x):
		norm = sqrt(sum(x**2))
		return norm
	
	#calculates the n-dimensional distance between two points
	def getDistance(self, x1, x2):
		dist = sqrt(sum((x1-x2)**2))
		return dist
	
	def getImageCorners(self, Im):
		corners = cv2.goodFeaturesToTrack(Im,200,0.01,10)
		return int0(corners)
	
	def getNeighMatrix(self, s, currX, currY, lx, ly):
		xs = [[(x,y) for x in range(currX-s+1,currX+s+1) if x>=0 if x < lx ] for y in range(currY-s+1,currY+s+1)  if y>=0 if y < ly]
		return xs
	
	#new functions for PA3:
		
	
		
