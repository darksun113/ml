class Perceptron(object):
	def __init__(self, input_num, activator):
		'''
		Initialize the Percetron. Configure the number of prameters and activator.
		The type of activator is double -> double
		'''

		self.activator = activator
		self.weights = [0.0 for _ in range(input_num)]
		self.bias = 0.0

	def __str__(self):
		'''
		Print weights and bias
		'''
		return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

	def predict(self, input_vec):
		'''
		Input vector. Predict and output the result from perceptron.
		'''
		#Pack input_vec[x1,x2,x3...] and weights[w1,w2,w3...] together
		#Now we have[(x1,w1),(x2,w2),(x3,w3)...]
		#We map() x1*w1, x2*w2, x3*w3 ...
		#Then reduce() to sum up
		return self.activator(
			reduce(lambda a,b: a+b,
				map(lambda(x,w): x*w,
					zip(input_vec, self.weights))
				, 0.0) + self.bias)

	def train(self, input_vecs, labels, iteration, rate):
		'''
		Input training data: a vector and a relevant label; number of iteration and learning rate.
		'''
		for i in range(iteration):
			self._one_iteration(input_vecs, labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		'''
		one iteration, traning set traversal
		'''
		#Pack input and output together, then we get a list [(input_vec, label)...]
		#Then an individual training sample is (input_vectors, labels)
		samples = zip(input_vecs, labels)
		#For each sample, update the weight according to the rule of perceptron
		for (input_vec, label) in samples:
			#compute the output under the current weight
			output = self.predict(input_vec)
			#update the weight
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		'''
		update the weight according to the rule
		'''
		delta = label - output
		self.weights = map(
			lambda(x,w): w+rate*delta*x,
			zip(input_vec, self.weights))
		#update bias
		self.bias += rate * delta

		#print self.weights
		#print self.bias

def f(x):
	'''
	define activator function f
	'''
	return 1 if x > 0 else 0

def get_training_dataset():
	'''
	Construct a truth table for AND operation 
	'''

	input_vecs = [[1,1], [0,0], [1,0], [0,1]]
	labels = [1, 0, 0, 0]
	return input_vecs, labels

def train_and_perceptron():
	'''
	Use the perceptron
	'''
	p = Perceptron(2,f)
	input_vecs, labels = get_training_dataset()
	p.train(input_vecs, labels, 10, 0.1)
	return p


if __name__ == '__main__':
	and_perception = train_and_perceptron()
	print and_perception

