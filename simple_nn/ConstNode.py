'''
The node always output 1 (It is used to compute bias W_b)
'''

class ConstNode(object):
	def __init__(self, layer_index, node_index):
		'''
		similar to class Node
		'''
		self.layer_index = layer_index
		self.node_index = node_index
		self.downtream = []
		self.output = 1

	def append_downstream_connection(self, conn):
		'''
		Add a node to downtream
		'''
		self.downtream.append(conn)

	def calc_hidden_layer_delta(self):
		downtream_delta = reduce(
			lambda ret, conn: ret + conn.downtream_node.delta*conn.weight, self.downtream, 0.0)
		self.delta = self.output * (1 - self.output) * downtream_delta

	def __str__(self):
        '''
        print node info
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
