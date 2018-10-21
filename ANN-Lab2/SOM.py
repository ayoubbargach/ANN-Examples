import os
import math
import numpy as np
import matplotlib.pyplot as plt

# We format the given data from animals.dat
os.chdir( os.path.dirname(os.path.abspath(__file__)) )

# Open data files
animalnames_f = open("data_lab2/animalnames.txt", "r")
animalnames = animalnames_f.read()
animalnames_f.close()

animalattr_f = open("data_lab2/animalattributes.txt", "r")
animalattr = animalattr_f.read()
animalattr_f.close()

matrix_f = open("data_lab2/animals.dat", "r")
matrix_a = matrix_f.read()
matrix_f.close()

#Format data
animals = animalnames.split("\r\n")
del animals[len(animals) - 1] # An empty element is added in the end, I do not know why ?
attr = animalattr.split("\r\n")

raw_matrix = matrix_a.split(",")

matrix = [ [ int( raw_matrix[j + i * len(attr)] ) for j in range(0, len(attr)) ] for i in range(0, len(animals)) ]
		

"""
# For printing some attr
for i in range( 0, len(animals)):
	print( "NEW ---------- " + animals[i] )
	for j in range( 0, len(attr)):
		if matrix[i][j] == 1:
			print( attr[j] )
"""

# HERE the data is ready --- matrix formated with 84 attributes in each line
# print( matrix )

#INITIALISATION
# We generate random weights for a matrix of 100x84 (84 attributes for each node)
nodes = 100
weights = []
low = 0
high = 1
epochs = 20
learning_rate = 0.2
learning_rate_n = 0.2

# Distance whithout the root to same computing time
def distance( x, w ):
	result = x - w
	for i in range( 0, len(x)):
		result[i] = result[i]**2
	return np.sum( result )

# Number of neighbourhoods parameter : Linear
class n_parameter:
	""" Class to manage Neighboorhood parameter which 
	decreases here linear against epochs """
	
	def __init__(self, maximum, minimum, total_epochs):
		self.maximum = maximum
		self.result = maximum
		self.minimum = minimum
		self.epoch = 0
		self.total = total_epochs - 1 # -1 to reach maximas

	def get_number(self):
		""" This function autoincrement epoch and return the number of neighbourhoods """
		self.result = int( math.ceil(- ( self.maximum - self.minimum ) * self.epoch / self.total + self.maximum) )
		self.epoch += 1
		return self.result


for i in range(0, nodes):
	weights.append(np.random.uniform(low, high, len(attr) ))


weights = np.asarray(weights)
matrix = np.asarray(matrix)

"""
# Print the weights for test :
print( weights[0].shape )
print( matrix[0].shape )
"""
# We create n_parameter class, use neightbours_parameter.get_number()
neightbours_parameter = n_parameter( 50, 1, epochs)

#TRAINING
for i in range(0, epochs):
	# Neighbours number
	n_number = neightbours_parameter.get_number()

	for j in range(0, len(animals)): # len(animalnames) is the number of points we have (inner loop)
		# New loop to calculate the distance :
		min_distance = distance( matrix[j], weights[0] )
		index = 0

		for k in range( 1, nodes) :
			result = distance( matrix[j], weights[k] )
			if min_distance > result :
				min_distance = result
				index = k
		

		
		# Once we have the index of the winner, we can update the weights of the winner and those of the neighbourhoods
		
		# Winner update
		weights[index] += learning_rate * (matrix[j] - weights[index])

		# Neighbourhoods update according to neightbours_parameter and learning_rate_n

		for k in range( index-n_number, index+n_number ):
			if k >= 0 and k < nodes :
				weights[k] += learning_rate_n * (matrix[j] - weights[k])

		# learning_rate_n can be a function that reduces against epochs
	

#PRINT
# this time we range the weight for each input to find the clothest one, and we save index
pos = []
for j in range(0, len(animals)):
	min_distance = distance( matrix[j], weights[0] )
	index = 0
	for k in range( 1, nodes) : 
		result = distance( matrix[j], weights[k] )
		if min_distance > result :
			min_distance = result
			index = k

	pos.append( (animals[j],index) )
	


# Sort the list to find similarities of animals
dtype = [('animal', 'S10'), ('i', int)]
sorted_array = np.array(pos, dtype=dtype)
sorted_array = np.sort(sorted_array, order='i') 

print( sorted_array )

















