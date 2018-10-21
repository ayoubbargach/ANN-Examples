# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# We format the given data from animals.dat
os.chdir( os.path.dirname(os.path.abspath(__file__)) )

# Open data file
matrix_f = open("data_lab2/cities.dat", "r")
matrix_a = matrix_f.read()
matrix_f.close()

#Format data
cities = matrix_a.split(";\r\n")
matrix = [ [ float(x) for x in cities[i].split(", ") ] for i in range(0, len(cities)) ]


# HERE the data is ready --- matrix formated with 84 attributes in each line
# print( matrix )

#INITIALISATION
# We generate random weights for a matrix of 100x84 (84 attributes for each node)
nodes = 10
weights = []
low = 0
high = 1
epochs = 100
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
	weights.append(np.random.uniform(low, high, len(matrix[0]) ))


weights = np.asarray(weights)
matrix = np.asarray(matrix)

"""
# Print the weights for test :
print( weights[0].shape )
print( matrix[0].shape )
"""

# We create n_parameter class, use neightbours_parameter.get_number()
neightbours_parameter = n_parameter( 1, 1, epochs)

#TRAINING
for i in range(0, epochs):
	# Neighbours number
	n_number = neightbours_parameter.get_number()

	for j in range(0, len(matrix)): # len(animalnames) is the number of points we have (inner loop)
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
			if k >= nodes: # cyclic tour
				k = k % nodes
			weights[k] += learning_rate_n * (matrix[j] - weights[k])

		# learning_rate_n can be a function that reduces against epochs

#PRINT
# this time we range the weight for each input to find the clothest one, and we save index
pos = []
X = []
Y = []
for j in range(0, len(matrix)):
	min_distance = distance( matrix[j], weights[0] )
	index = 0
	for k in range( 1, nodes) :
		result = distance( matrix[j], weights[k] )
		if min_distance > result :
			min_distance = result
			index = k

	pos.append( (j,index) )
	X.append( matrix[j][0] )
	Y.append( matrix[j][1] )
	


# Sort the list to find similarities of animals
dtype = [('city', int), ('i', int)]
sorted_array = np.array(pos, dtype=dtype)
sorted_array = np.sort(sorted_array, order='i')

print( sorted_array )

# Plot section
for i in range(0, len(matrix) - 1):
	city1 = sorted_array[i][0]
	city2 = sorted_array[i+1][0]
   	plt.plot( [X[ city1 ],X[ city2 ]] , [Y[ city1 ],Y[ city2 ]] , 'ro-')

city1 = sorted_array[len(matrix) - 1][0]
city2 = sorted_array[0][0]
plt.plot( [X[ city1 ],X[ city2 ]] , [Y[ city1 ],Y[ city2 ]] , 'ro-') # Final point

plt.show()












