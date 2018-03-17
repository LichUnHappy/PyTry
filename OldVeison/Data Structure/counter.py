class Counter(object):
	"""
	Models a counter.
	"""

	#class variable
	instances = 0

	#Constructor
	def __init__(self):
		"""Set up the counter"""
		Counter.instances += 1
		self.reset()	

	#Mutator methods
	def reset(self):
		"""Set the counter to 0"""
		self._value = 0

	def increment(self, amount = 1):
		"""Add amount to the counter"""
		self._value += amount

	def decrement(self, amount = 1):
		"""Subtract amount from the counter """
		self._value -= amount

	#Accessor methods
	def getValue(self):
		"""Return the counter's value"""
		return self._value

	def __str__(self):
		"""Return the string representation of the counter """
		return str(self._value)

	def __eq__(self, other):
		"""Return True if self equals other or False otherwise"""
		if self is other: return True
		if type(self) != type(other): return False
		return self._value == other._value
