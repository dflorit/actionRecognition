from util import Util
from numpy import *
from math import *
from cv2 import *
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import *
    from sklearn.neural_network import MLPClassifier

class SVCClassifier:
	
	def __init__(self, k = 'linear', deg = 3):
		self.u = Util()
		self.k = k
		self.d = deg
		
		self.s = svm.SVC(kernel= self.k, degree = self.d)
		self.parameters = self.s.get_params()
		
	def training(self, X, y):
		self.s.fit(X, y)
		
	def test(self, X):
		return self.s.predict(X)
		
	def getParameters(self):
		return self.parameters
		
class SVCLinearClassifier:
	
	def __init__(self):
		self.u = Util()
		self.s = svm.LinearSVC()
		self.parameters = self.s.get_params()
		
	def training(self, X, y):
		self.s.fit(X, y)
		
	def test(self, X):
		return self.s.predict(X)
		
	def getParameters(self):
		return self.parameters
		

class NNClassifier:
	
	def __init__(self, nnAlpha = 1e-5, nnHiddenLayersNumber = 15):
		self.u = Util()
		self.a = nnAlpha
		self.hh = nnHiddenLayersNumber
		self.n = MLPClassifier(solver='lbgfs', alpha= self.a, hidden_layer_sizes=(self.hh,), random_state=1)
		
	def training(self, X, y):
		self.n.fit(X, y)
		
	def test(self, X):
		return self.n.predict(X)
		
