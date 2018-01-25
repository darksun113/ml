from perceptron import Perceptron

f = lambda x: x

class LinearUnit(Perceptron):
	def __init__(self, input_num):
		'''
		Initialize LinearUnit and set the number of feature
		'''
		Perceptron.__init__(self, input_num, f)

def get_training_dateset():
	'''
	Contruct a traning sample with 5 persons
	'''
	#Input is a vectors of working years for each person
	input_vecs = [[5],[3],[8],[1.4],[10.1]]
	#labels
	labels = [5500,2300,7600,1800,11400]
	return input_vecs, labels

def train_linear_unit():
	'''
	User data to train Linear Unit
	'''
	#only consider 1 feature
	lu = LinearUnit(1)
	#train 10 iterartions, learning rate sets to 0.01
	input_vecs, labels = get_training_dateset()
	lu.train(input_vecs, labels, 10, 0.01)
	return lu

if __name__ == '__main__':
	'''Train Linear Unit'''
	linear_unit = train_linear_unit()
	# Print trained weight
	print linear_unit

	#test and predict
	print linear_unit.predict([3.4])
	print linear_unit.predict([5])