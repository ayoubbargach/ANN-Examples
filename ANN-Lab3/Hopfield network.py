import numpy as np

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

# Remove duplicate patterns
def remove_duplicates(input):
	for i in range(0, len(input)):
		for j in range (i + 1, len(input)):
			if(count_errors(input[i], input[j]) == 0):
				input = np.delete(input, j, 0)
				input = remove_duplicates(input)
				return input
	return input

# Memory pattern initiation
memory_patterns = []
memory_patterns.append([-1, -1, 1, -1, 1, -1, -1, 1])            #x1
memory_patterns.append([-1, -1, -1, -1, -1, 1, -1, -1])          #x2
memory_patterns.append([-1, 1, 1, -1, -1, 1, -1, 1])             #x3
memory_patterns = np.array(memory_patterns)
print("TRAINING PATTERNS")
print(memory_patterns)
print("")

# Weight initiation
weights = np.matmul(memory_patterns.T, memory_patterns)
# Remove self connections
#for i in range (0, len(weights)):
	#weights[i][i] = 0
print("WEIGHTS")
print(weights)
print("")

# Distorted pattern initiation
x1d = [1, -1, 1, -1, 1, -1, -1, 1]
x2d = [1, 1, -1, -1, -1, 1, -1, -1]
x3d = [1, 1, 1, -1, 1, 1, -1, 1]

'''
# Search all possible input patterns for attractors
all_possible_input_patterns = []
get_all_possible_inputs(all_possible_input_patterns, [], 8, 0)
attractors = []

for input in all_possible_input_patterns:
	old_input_pattern = input
	input_pattern = sgn(np.dot(weights, old_input_pattern))
	count = 1
	while (count < 100): #(count_errors(input_pattern, old_input_pattern) != 0)
		old_input_pattern = input_pattern
		input_pattern = sgn(np.dot(weights, old_input_pattern))
		count = count + 1
		if(count_errors(input_pattern, old_input_pattern) == 0):
			attractors.append(input_pattern)
			break
			
attractors = remove_duplicates(attractors)
print("ATTRACTORS: ", len(attractors))
print(attractors)
'''


# Test single pattern
old_input_pattern = np.copy(x3d)
print("ORIGINAL INPUT PATTERN")
print(old_input_pattern)
print("DIFFERENCE TO ORIGINAL PATTERNS")
print(count_errors(old_input_pattern, memory_patterns[0]))
print(count_errors(old_input_pattern, memory_patterns[1]))
print(count_errors(old_input_pattern, memory_patterns[2]))
input_pattern = sgn(np.dot(weights, old_input_pattern))
count = 1
print("")
print("Count: ", count)
print("CURRENT PATTERN")
print(input_pattern)
print("DIFFERENCE TO ORIGINAL PATTERNS")
print(count_errors(input_pattern, memory_patterns[0]))
print(count_errors(input_pattern, memory_patterns[1]))
print(count_errors(input_pattern, memory_patterns[2]))
#while ((count_errors(input_pattern, old_input_pattern) != 0)):
while (count < 5):
	old_input_pattern = np.copy(input_pattern)
	input_pattern = sgn(np.dot(weights, old_input_pattern))
	count = count + 1
	print("")
	print("Count: ", count)
	print("CURRENT PATTERN")
	print(input_pattern)
	print("DIFFERENCE TO ORIGINAL PATTERNS")
	print(count_errors(input_pattern, memory_patterns[0]))
	print(count_errors(input_pattern, memory_patterns[1]))
	print(count_errors(input_pattern, memory_patterns[2]))

	
