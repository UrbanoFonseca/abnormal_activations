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
		return (1.7159 * np.tanh(2 * x / 3))


	def ReSech(x):
		# As presented in 'A novel activation for multilayer feed-forward neural networks'
		# from Njikam, A.B.S. and Zhao, H.
		# where sech is the hyperbolic secant
		sech_x = 1 / np.cosh(x)
		return x * sech_x


	def scaled_sigmoid(x):
		# As presented in 'Revise Saturated Activation Functions'
		# by Xu, B, and Huang, R. and Li, M.
		return 4 / ( 1 + np.exp(-x)) - 2


	def penalized_tanh(x, alpha=0.25):
		# As presented in 'Revise Saturated Activation Functions'
		# by Xu, B, and Huang, R. and Li, M.
		return np.tanh(x) if x > 0 else alpha * np.tanh(x)

	
	def trunc_sin(x):
		# Based on the 'Taming the Waves:
		# Sine as Activation Function in Deep Neural Networks'
		# from Parascandolo, G. and Huttunen, H. and Virtanen, T.
		if x < -np.pi/2:
			return 0
		elif x > np.pi/2:
			return 1
		else:
			return np.sin(x)

	def sin(x):
		return np.sin(x)