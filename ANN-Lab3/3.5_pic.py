import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle
import os
import math

np.set_printoptions(threshold=np.nan) #Always print the whole matrix
random.seed()

# To add noise knowing a parameter p (percentage of noise)
def add_noise( pattern, p ) :
	order = list(range(len(pattern)))
	shuffle( order )
	part = int(len(order) * (p/100))
	for i in order[:part] :
		pattern[i] *= -1
	return pattern

# Count how many bits differ between pattern x and y
def count_errors(x, y):
	count = 0
	for i in range(0, len(x)):
		if (x[i] != y[i]):
			count = count + 1
	return(count)

# Set any value in x >= 0 to 1 and any value < 0 to -1
def sgn(x):
	for i in range(0, len(x)):
		if (x[i] >= 0):
			x[i] = 1
		else:
			x[i] = -1
	return(x)

# Generate all possible input patterns of the chosen size
def get_all_possible_inputs(input, sub_part, size, depth):
	if(size == depth + 1):
		input.append(sub_part + [1])
		input.append(sub_part + [-1])
		return
	get_all_possible_inputs(input, sub_part + [1], size, depth + 1)
	get_all_possible_inputs(input, sub_part + [-1], size, depth + 1)

# Generate random pattern of the given size
def generate_random_pattern(size):
	pattern = [0] * size
	for i in range(0, size):
			if(random.randint(0, 1) == 1):
			#if(sgn(np.random.normal(0.5, 1, 1)) >= 0):
				pattern[i] = 1
			else:
				pattern[i] = -1
	return pattern

# Remove duplicate patterns
def remove_duplicates(input):
	for i in range(0, len(input)):
		for j in range (i + 1, len(input)):
			if(count_errors(input[i], input[j]) == 0):
				input = np.delete(input, j, 0)
				input = remove_duplicates(input)
				return input
	return input
	
def simple_add_noise(pattern, size_of_noise):
	for i in range(0, size_of_noise):
		row = random.randint(0, np.size(pattern, 0) - 1)
		column = random.randint(0, np.size(pattern, 1) - 1)
		pattern[row][column] = pattern[row][column] * -1
	return pattern
	
def simple_even_add_noise(pattern, size_of_noise):
	for row in range(np.size(pattern, 0) - 1):
		for i in range(0, size_of_noise):
			column = random.randint(0, np.size(pattern, 1) - 1)
			pattern[row][column] = pattern[row][column] * -1
	return pattern

def another_simple_add_noise(pattern, size_of_noise):
	for row in range(np.size(pattern, 0)):
		for i in range(0, size_of_noise):
			pattern[row][i] = pattern[row][i] * -1
	return pattern

#____________________________________________________
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
epochs = 10
pict_def = 32
number_patterns = 11
number_nodes = 300

# Figure constants
rows = 3
columns = 3

# pict = [ [ [ int( raw_pict[j + i * pict_def + k * number_patterns] ) for j in range(0, pict_def) ] for i in range(0, pict_def) ] for k in range(0, number_patterns) ]
pict = [ [ int( raw_pict[j  + i * number_nodes] ) for j in range(0, number_nodes) ] for i in range(0, number_patterns) ]
#____________________________________________________

# TO TEST RANDOM PATTERNS
# Generate 300 random patterns
pattern_length = 100
patterns_to_generate = 300

random_patterns = generate_random_pattern(pattern_length) # Needs to be instantiated with a pattern of same length as other random patterns so that vstack won't complain about dimensions
random_patterns = np.vstack([random_patterns, generate_random_pattern(pattern_length)])
while (np.size(random_patterns, 0) < patterns_to_generate): # If there is only one row in random_patterns this will instead use the size of the columns in that row
	random_patterns = np.vstack([random_patterns, generate_random_pattern(pattern_length)])
	if (np.size(random_patterns, 0) == patterns_to_generate):
		random_patterns = remove_duplicates(random_patterns)
		
noisy_patterns = np.copy(random_patterns)
for i in range(0, patterns_to_generate):
	noisy_patterns[i] = add_noise(noisy_patterns[i], 5)


'''
#TO TEST PICS
random_patterns = np.array(pict)
#With noise
print(len(random_patterns))
noisy_patterns = np.copy(random_patterns)
for i in range(0, 11):
	noisy_patterns[i] = add_noise(noisy_patterns[i], 0)
'''

convergences = []
for p in range (1, np.size(noisy_patterns, 0)):
	sub_pattern = np.copy(random_patterns[:p,:])
	weights = np.matmul(sub_pattern.T, sub_pattern)
	# Remove self connections
	for i in range (0, np.size(weights, 0)):
		weights[i][i] = 0
	#noisy_sub_pattern = np.copy(noisy_patterns[:p,:])
	convergence = 0
	for i in range (0, p):
		old_input_pattern = noisy_patterns[i]
		input_pattern = sgn(np.dot(weights, old_input_pattern))
		count = 0
		while (count < 25):
			old_input_pattern = np.copy(input_pattern)
			input_pattern = sgn(np.dot(weights, old_input_pattern))
			if(count_errors(input_pattern, old_input_pattern) == 0):
				if(count_errors(input_pattern, random_patterns[i]) == 0):
					convergence = convergence + 1
				break
			count = count + 1
	convergences.append(convergence)
	print("Iteration: ", p, "converging patterns: ", convergence)

'''
for i in range(0, 5):
	for j in range(0, 5):
		print("I:", i, "J:", j, "DIFFERENCE:", count_errors(random_patterns[i], random_patterns[j]))
'''

ax = plt.gca()
ax.plot(convergences)
plt.ylabel('Number of stable patterns')
plt.xlabel('Patterns attempted to be memorized')
plt.show()
'''
for i in range (0, 299):
	convergences[i] = float(convergences[i])/(i + 1)
ax = plt.gca()
ax.plot(convergences)
plt.ylabel('Ratio of patterns actually memorized')
plt.xlabel('Patterns attempted to be memorized')
plt.show()
'''