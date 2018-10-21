import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan) #Always print the whole matrix
random.seed()

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
	
# Adds noise to pattern
def add_noise(pattern, size_of_noise):
	if(size_of_noise > np.size(pattern, 0)):
		size_of_noise = np.size(pattern, 0) * np.size(pattern, 1)
	
	#print(np.size(pattern, 0) - 1)
	#print(np.size(pattern, 1) - 1)
	noise_loc = [random.randint(0, np.size(pattern, 0) - 1), random.randint(0, np.size(pattern, 1) - 1)]
	if (size_of_noise == 1): #|| np.size(pattern, 0) - 1 == 1
		pattern[noise_loc[0]][noise_loc[1]] = pattern[noise_loc[0]][noise_loc[1]] * -1
	else:
		noise_loc = np.vstack([noise_loc, [random.randint(0, np.size(pattern, 0) - 1), random.randint(0, np.size(pattern, 1) - 1)]])
		while(np.size(noise_loc, 0) < size_of_noise):
			noise_loc = np.vstack([noise_loc, [random.randint(0, np.size(pattern, 0) - 1), random.randint(0, np.size(pattern, 1) - 1)]])
			noise_loc = remove_duplicates(noise_loc)
			print(np.size(noise_loc, 0))
		for i in range(0, np.size(noise_loc, 0)):
			pattern[noise_loc[i][0]][noise_loc[i][1]] = pattern[noise_loc[i][0]][noise_loc[i][1]] * -1
	return pattern
	
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

# Generate 300 random patterns
pattern_length = 100
patterns_to_generate = 300
random_patterns = generate_random_pattern(pattern_length) # Needs to be instantiated with a pattern of same length as other random patterns so that vstack won't complain about dimensions
while (np.size(random_patterns, 0) < patterns_to_generate): # If there is only one row in random_patterns this will instead use the size of the columns in that row
	random_patterns = np.vstack([random_patterns, generate_random_pattern(pattern_length)])
	if (np.size(random_patterns, 0) == patterns_to_generate):
		random_patterns = remove_duplicates(random_patterns)
	
'''
#Without noise
# Add one pattern to be memorized at a time and check how many of the stored patterns are stable
stability = []
for p in range (1, np.size(random_patterns, 0)):
	# Weight initiation
	sub_pattern = np.copy(random_patterns[:p,:])
	weights = np.matmul(sub_pattern.T, sub_pattern)
	# Remove self connections
	for i in range (0, np.size(weights, 0)):
		weights[i][i] = 0

	#sub_pattern = simple_even_add_noise(sub_pattern, 5)
	# Check number of stable patterns
	number_of_stable_patterns = 0
	for i in range (0, np.size(sub_pattern, 0)):
		pattern = sgn(np.dot(weights, sub_pattern[i]))
		if(count_errors(pattern, random_patterns[i]) == 0):
			number_of_stable_patterns = number_of_stable_patterns + 1

	stability.append(number_of_stable_patterns)
	print("Iteration: ", p, "stable patterns: ", number_of_stable_patterns)


ax = plt.gca()
ax.plot(stability)
plt.title('Number of stable patterns for network with biased patterns without self connections and without noise')
plt.ylabel('Number of stable patterns')
plt.xlabel('Patterns memorized')
plt.show()

for i in range (1, 299):
	print(stability[i], i)
	stability[i] = float(stability[i])/(i + 1)
ax = plt.gca()
ax.plot(stability)
plt.title('Ratio of successfully memorized patterns for self connected network without noise')
ax.legend(['Self connected without noise'])
plt.ylabel('Ratio of patterns actually memorized')
plt.xlabel('Patterns attempted to be memorized')
plt.show()
'''


#With noise
noisy_patterns = np.copy(random_patterns)
noisy_patterns = simple_even_add_noise(noisy_patterns, 5)
convergences = []
for p in range (1, np.size(noisy_patterns, 0)):
	print(p)
	sub_pattern = np.copy(random_patterns[:p,:])
	weights = np.matmul(sub_pattern.T, sub_pattern)
	# Remove self connections
	#for i in range (0, np.size(weights, 0)):
		#weights[i][i] = 0
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

ax = plt.gca()
ax.plot(convergences)
plt.ylabel('Number of stable patterns')
plt.xlabel('Patterns attempted to be memorized')
plt.show()

for i in range (0, 299):
	convergences[i] = float(convergences[i])/(i + 1)
ax = plt.gca()
ax.plot(convergences)
plt.ylabel('Ratio of patterns actually memorized')
plt.xlabel('Patterns attempted to be memorized')
plt.show()
