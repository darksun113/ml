#Node class, record and mentain self information, and the relevant connections to upsteam or downstream. 
#It also computes the output and delta.

class Node(object):
	'''
	Contruct a object of Node
	layer_index: the index of the layer that the node belongs to
	node_index: the index of node
	'''
	def __init__(self, layer_index, node_index):
		self.layer_index = layer_index
		self.node_index = node_index
		self.downstream = []
		self.upsteam = []
		self.output = 0
		self.delta = 0

	def set_output(self, output):
		'''
		Set the output of the current node if the node belongs to input layer
		'''
		self.output = output

	def append_downstream_connection(self, conn):
		'''
		add a node from downstream
		'''
		self.downstream.append(conn)

	def append_upstream_connection(self, conn):
		'''
		add a node from upsteam
		'''
		self.upsteam.append(conn)

	def calc_output(self):
		'''
		compute the output for this node
		'''
		output = reduce(lambda ret, conn: ret + conn.upsteam_node.output * conn.weight, self.upsteam,0)
		self.output = sigmode(output)

	def calc_hidden_layer_delta(self):
		'''
		If a node belongs to hidden layer, compute delta
		'''
		downstream_delta = reduce(
			lambda ret, conn: ret + conn.downstream_node.delta*conn.weight, self.downstream, 0.0)
		self.delta = self.output * (1 - self.output) * downstream_delta

	def cal_output_layer_delta(self, label):
		'''
		If a node belongs to output layer, compute delta
		'''
		self.delta = self.output * (1 - self.output) * (label - self.output)

	def __str__(self):
		'''
		Print info of node
		'''
		node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		upsteam_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upsteam, '')
		return node_str + '\n\tdownstream:' + downstream_str + '\n\tinputstream:' + upsteam_str

