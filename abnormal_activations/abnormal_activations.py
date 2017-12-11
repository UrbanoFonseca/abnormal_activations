import numpy as np
from keras import backend as K

class ActivationFunctions():

	def alpha_linear(x, alpha=0.5):
		# The linear function Y = A * X
		return	alpha * x


	def step(x, threshold=0.0):
		# The step function returns 0 for x < threshold
		# and 1 otherwise.
		return 1 if x > threshold else 0


	def LeCunSigmoid(x, alpha=0.01):
		# As presented in the 'Generalization and Network Design Strategies'
		# from Y. LeCun
		# The alpha parameter is supposed to be a small linear term to avoid
		# flat spots.
		# f(x) = 1.7159 * tanh(2/3*x) + a*x
		return (1.7159 * K.tanh(2 * x / 3) + alpha * x)


	def ReSech(x):
		# As presented in 'A novel activation for multilayer feed-forward neural networks'
		# from Njikam, A.B.S. and Zhao, H.
		# where sech is the hyperbolic secant
		cosh = (K.exp(x)+K.exp(-x))/2
		sech_x = 1 / cosh
		return x * sech_x


	def scaled_sigmoid(x):
		# As presented in 'Revise Saturated Activation Functions'
		# by Xu, B, and Huang, R. and Li, M.
		return 4 / ( 1 + K.exp(-x)) - 2


	def penalized_tanh(x, alpha=0.25):
		# As presented in 'Revise Saturated Activation Functions'
		# by Xu, B, and Huang, R. and Li, M.
		return K.switch(x > K.variable(0), K.tanh(x), alpha * K.tanh(x))

	
	def trunc_sin(x):
		# Based on the 'Taming the Waves:
		# Sine as Activation Function in Deep Neural Networks'
		# from Parascandolo, G. and Huttunen, H. and Virtanen, T.
		pi = 3.14159265359
		return (K.switch(x < K.constant(-pi/2), K.constant(0), K.switch(x > K.constant(pi/2), K.constant(1), K.sin(x))))


	def sin(x):
		return K.sin(x)
