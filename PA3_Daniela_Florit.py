from util import Util
from numpy import *
from functions import *
import cv2

class Assignment3:
	
	def __init__(self):
		self.u = Util()
	
	def problem1a(self):
		p1 = "Images/p1/"
		
		#basketball image
		img1_ini = p1 + "basketball1.png"
		img1_end = p1 + "basketball2.png"
		img1_output = p1 + "basketballOutput.png"
		
		Im1_1 = cv2.imread(img1_ini)
		Im1_2 = cv2.imread(img1_end)
		Image1_1 = cv2.cvtColor(Im1_1, cv2.COLOR_BGR2GRAY)
		Image1_2 = cv2.cvtColor(Im1_2, cv2.COLOR_BGR2GRAY)
		Image1RGB = self.u.readImgRGB(img1_ini)
			
		lk_img1 = LucasKanade(Image1_1, Image1_2, Image1RGB)
		output1 = lk_img1.getSparseFlow()
		output1.save(img1_output)
		
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PA3 = Assignment3()


