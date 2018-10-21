# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# We format the given data from animals.dat
os.chdir( os.path.dirname(os.path.abspath(__file__)) )

# Open data file

# MPs data
mpnames_f = open("data_lab2/mpnames.txt", "r")
mpnames_a = mpnames_f.read()
mpnames_f.close()

mpparty_f = open("data_lab2/mpparty.dat", "r")
mpparty_a = mpparty_f.read()
mpparty_f.close()

mpsex_f = open("data_lab2/mpsex.dat", "r")
mpsex_a = mpsex_f.read()
mpsex_f.close()

mpdistrict_f = open("data_lab2/mpdistrict.dat", "r")
mpdistrict_a = mpdistrict_f.read()
mpdistrict_f.close()

# Votes data
votes_f = open("data_lab2/votes.dat", "r")
votes_a = votes_f.read()
votes_f.close()

#Format data
mpnames = mpnames_a.split("\n")
del( mpnames[len(mpnames) - 1] )


# % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
# % Use some color scheme for these different groups
mpparty = mpparty_a.split("\n\t")

# % Coding: Male 0, Female 1
mpsex = mpsex_a.split("\n\t")
mpdistrict = mpdistrict_a.split("\n\t")

# Votes format
raw_votes = votes_a.split(",")

# Build matrices
matrix_MPs = [ [ mpnames[i], int(mpparty[i]), int(mpsex[i]), int(mpdistrict[i]) ] for i in range(0, len(mpnames)) ]

# data of votes
votes = [ [ float( raw_votes[j + i * 31] ) for j in range(0, 31) ] for i in range(0, len(matrix_MPs)) ]


# HERE the data is ready --- matrix formated with 84 attributes in each line
# print( matrix )

#INITIALISATION
# We generate random weights for a matrix of 100x84 (84 attributes for each node)
nodes = (10, 10) # 2-dimensionnal grid
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


for i in range(0, nodes[0]):
	weights.append( [] )
	for j in range(0, nodes[1]):
		weights[i].append(np.random.uniform(low, high, len(votes[0]) ))


weights = np.asarray(weights)
votes = np.asarray(votes)

print( votes )

"""
# Print the weights for test :
print( weights.shape )
"""

# We create n_parameter class, use neightbours_parameter.get_number()
neightbours_parameter = n_parameter( 0, 1, epochs)

#TRAINING
for l in range(0, epochs):
	# Neighbours number
	n_number = neightbours_parameter.get_number()

	for m in range(0, len(matrix_MPs)):
		# New loop to calculate the distance :
		min_distance = distance( votes[m], weights[0][0] )
		index = (0, 0)
		

		for i in range( 0, nodes[0]) :
			for j in range( 0, nodes[1]) :
				result = distance( votes[m], weights[i][j] ) # weights is 3D
				if min_distance > result :
					min_distance = result
					index = (i, j)
		"""
		print( index )
		print( "FOUND" )	
		"""
		
		# Once we have the index of the winner, we can update the weights of the winner and those of the neighbourhoods
		
		# Winner update
		weights[index[0],index[1]] += learning_rate * (votes[m] - weights[index[0],index[1]])

		# Neighbourhoods update according to neightbours_parameter and learning_rate_n
		for r in range( index[0]-n_number, index[0]+n_number ):
			if r >= 0 and r < nodes[1]: # Not a cyclic update
				for k in range( index[1]-n_number, index[1]+n_number ): # We start by updating row neighbours
					if k >= 0 and k < nodes[0]: # Not a cyclic update
						weights[k] += learning_rate_n * (votes[m] - weights[r][k])

			

		# learning_rate_n can be a function that reduces against epochs

#PRINT
# this time we range the weight for each input to find the clothest one, and we save index
pos = []
for m in range(0, len(matrix_MPs)):
	min_distance = distance( votes[m], weights[0][0] )
	index = (0, 0)

	for i in range( 0, nodes[0]) :
		for j in range( 0, nodes[1]) :
			result = distance( votes[m], weights[i][j] )
			if min_distance > result :
				min_distance = result
				index = (i, j)

	pos.append( (matrix_MPs[m],index) )
	

"""
# Sort the list to find similarities of animals
 dtype = [('j', int), ('i', int)]
sorted_array = np.array(pos, dtype=dtype)
sorted_array = np.sort(sorted_array, order='i')
print( sorted_array )

"""


# Plot the results
print( pos )

# ------- INIT 

"""
sort_party = []
for i in range( 0, 8 ):
	sort_party.append( [] )
	sort_party[i].append( [] )
	sort_party[i].append( [] )

# % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
for m in range( 0, len( pos )):
	sort_party[ pos[m][0][1] ][0].append( pos[m][1][0] ) # pos[m][1][0] is the int for the party !!
	sort_party[ pos[m][0][1] ][1].append( pos[m][1][1] )

color=iter(cm.rainbow(np.linspace(0,1,8)))
print( sort_party )

m='o'

for i in range(0, 8):
	c=next(color)
	xs = sort_party[i][0]
	ys = sort_party[i][1]
	zs = i
	ax.scatter(xs, ys, zs, c=c, marker=m)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Each party')
"""
"""
sort_gender = []
for i in range( 0, 2 ):
	sort_gender.append( [] )
	sort_gender[i].append( [] )
	sort_gender[i].append( [] )

# % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
for m in range( 0, len( pos )):
	sort_gender[ pos[m][0][2] ][0].append( pos[m][1][0] ) # pos[m][1][0] is the int for the party !!
	sort_gender[ pos[m][0][2] ][1].append( pos[m][1][1] )

color=iter(cm.rainbow(np.linspace(0,1,8)))
print( sort_gender )

m='o'

for i in range(0, 2):
	c=next(color)
	xs = sort_gender[i][0]
	ys = sort_gender[i][1]
	zs = i
	ax.scatter(xs, ys, zs, c=c, marker=m)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Gender (0 Male/ 1 Female)')
"""

sort_district = []
for i in range( 0, 30 ):
	sort_district.append( [] )
	sort_district[i].append( [] )
	sort_district[i].append( [] )

# % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
for m in range( 0, len( pos )):
	sort_district[ pos[m][0][3] ][0].append( pos[m][1][0] ) # pos[m][1][0] is the int for the party !!
	sort_district[ pos[m][0][3] ][1].append( pos[m][1][1] )

color=iter(cm.rainbow(np.linspace(0,1,30)))
print( sort_district )

m='o'

for i in range(0, 30):
	c=next(color)
	xs = sort_district[i][0]
	ys = sort_district[i][1]
	zs = i
	ax.scatter(xs, ys, zs, c=c, marker=m)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('District')



plt.show()



