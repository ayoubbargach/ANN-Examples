# -*- coding: utf-8 -*-
from __future__ import division
from random import shuffle
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Some useful functions

# Set any value in x >= 0 to 1 and any value < 0 to -1
def sgn(x):
	if (x >= 0):
		x = 1
	else:
		x = -1
	return(x)

# To add noise knowing a parameter p (percentage of noise)
def add_noise( pattern, p ) :
	order = range( 0, len( pattern ))
	shuffle( order )
	part = int(len(order) * (p/100))
	for i in order[:part] :
		pattern[i] *= -1
	return pattern

# To update the weights
# As the matrix is symetric with 0 in the diagonal, we will only fill one triangle
def weights_update_origin(weights, patterns) :
	for i in range( 0, len( weights )) :
		for j in range( i , len( weights )) :
			result = 0
			# We take the first pattern dimension, in the begun case, it was 32 lines (in 2D pict).
			dim = len( patterns[0] )
			for k in range( 0, len( patterns )):
				result += patterns[k][i // dim][ i % dim ] * patterns[k][j // dim][ j % dim ]
			weights[i][j] = ( 1 / len( weights ) ) * result
			# The matrix is symetric, we update also the other triangle
			weights[j][i] = ( 1 / len( weights ) ) * result
	# print( weights )
	return weights

def weights_update(patterns, number) :
	return np.matmul(patterns[:number].T, patterns[:number])

# To generate the next pattern from the weights
def patterns_update( weights, patterns ):
	for i in range( 0, len( patterns ) ):
		save_pattern = patterns[i]
		result = 0
		dim = len( patterns[0] )
		for l in range( 0, len( weights )) :
			for m in range( 0, len( weights )) :
				result += weights[l][m] * save_pattern[ m // dim][ m % dim ]
			patterns[i][ l // dim][ l % dim ] = sgn( result )
	return patterns

# To generate the next pattern from the weights : ORGINAL Hopfield !
def get_pattern( weights, pattern ):
	order = range( 0, len( pattern ))
	# Randomize the entry sequence, the result differs completely !!
	shuffle( order )
	for l in order :
		result = 0
		for m in range( 0, len( weights )) :
			result += weights[l][m] * pattern[ m ]
		pattern[l] = sgn( result )
	return pattern
	

# We format the given data from animals.dat
os.chdir( os.path.dirname(os.path.abspath(__file__)) )

# Open data files
pict_f = open("pict.dat", "r")
pict_a = pict_f.read()
pict_f.close()

raw_pict = pict_a.split(",")

#print("RAW DATA LENGTH")
#print( len(raw_pict) )

# Some Constants
epochs = 2
pict_def = 32
number_patterns = 11
number_nodes = 1024

# Figure constants
rows = 3
columns = 3


# Plot print : 
fig = plt.figure()

# pict = [ [ [ int( raw_pict[j + i * pict_def + k * number_patterns] ) for j in range(0, pict_def) ] for i in range(0, pict_def) ] for k in range(0, number_patterns) ]
pict = [ [ int( raw_pict[j  + i * number_nodes] ) for j in range(0, number_nodes) ] for i in range(0, number_patterns) ]

# print( pict )
# plt.imshow(pict[3])


plot_save_pict = []
plot_save_pict2 = []
plot_pict = []


# HERE the data is ready --- 9 patterns of 32x32 pict matrix
# INIT the weights matrix for N nodes, so we have N^2 weights
weights = np.zeros( (number_nodes, number_nodes) , dtype = float)

# LEARNING
# convert
pict = np.array( pict )
weights = weights_update( pict, 3 )

save_pict = pict

# Plot format
for i in range( 0, len(pict) ) :
	plot_save_pict.append( [ [ int( save_pict[i][j  + k * pict_def] ) for j in range(0, pict_def) ] for k in range(0, pict_def) ] )

fig.add_subplot(rows, columns,1)
plt.imshow(plot_save_pict[0])
fig.add_subplot(rows, columns,2)
plt.imshow(plot_save_pict[1])
fig.add_subplot(rows, columns,3)
plt.imshow(plot_save_pict[2])

# We add the noise
for i in range( 0, number_patterns) :
	pict[i] = add_noise( pict[i], 90 )

save_pict2 = pict


for i in range( 0, len(pict) ) :
	plot_save_pict2.append( [ [ int( save_pict2[i][j  + k * pict_def] ) for j in range(0, pict_def) ] for k in range(0, pict_def) ] )

fig.add_subplot(rows, columns,4)
plt.imshow(plot_save_pict2[0])
fig.add_subplot(rows, columns,5)
plt.imshow(plot_save_pict2[1])
fig.add_subplot(rows, columns,6)
plt.imshow(plot_save_pict2[2])



# We udpate the weights with the inital patterns
# weights = weights_update(weights, pict)



"""
# diagonal of 0
print( weights.shape )
for i in range (0, len(weights)):
	weights[i][i] = 0
"""

# First CALL
for e in range( 0 , epochs ) :
	for i in range( 0, 3 ) :
		pict[i] = get_pattern(weights, pict[i])


# TESTING
"""
# Pattern stability : Here we verify if the patterns are stable :
if np.array_equal(save_pict[:3], pict[:3]) == True :  # test if same shape, same elements values, only for the 3 first
	print( "The patterns are stable !" )
else :
	print( "UNSTABLE" )
"""

for i in range( 0, len(pict) ) :
	plot_pict.append( [ [ int( pict[i][j  + k * pict_def] ) for j in range(0, pict_def) ] for k in range(0, pict_def) ] )



"""
# Stable table
for i in range( 1 , 4 ) : # (rows*columns//2) + 1
	fig.add_subplot(rows, columns, i*2 - 1)
	plt.imshow(plot_save_pict[i - 1])

for i in range( 1 , 4 ) :
	fig.add_subplot(rows, columns, i*2)
	# print( save_pict[i] )
	plt.imshow( plot_pict[i - 1] )
"""

# Degraded patterns




fig.add_subplot(rows, columns,7)
plt.imshow( plot_pict[0] )
fig.add_subplot(rows, columns,8)
plt.imshow( plot_pict[1] )
fig.add_subplot(rows, columns,9)
plt.imshow( plot_pict[2] )





# print( pict[0] )
# print( weights )
# print(save_pict[0])

plt.show()

