from abc import ABC, abstractclassmethod

class Membership(ABC):
	@abstractclassmethod
	def fuzzification():
		pass

	def defuzzification():
		pass