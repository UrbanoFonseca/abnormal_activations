class NameActivation(Activation):
	# This classes expands the vectorized functions to have a name that can
	# be propered passed and saved on the ModelCheckpoint callback.
	# https://github.com/fchollet/keras/issues/8716
	def __init__(self, custom_activation, name, **kwargs):
		super(NameActivation, self).__init__(custom_activation, **kwargs)
		self.__name__ = name