import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

x_lower_interval = 0
x_upper_interval = 2*math.pi
y_lower_interval = -2
y_upper_interval = 2
step_length = 0.1

#learning_rate = 0.01

random.seed(a=None)

# Class that represents a 2d input and what type it should be classified as
class Node:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.variance = v
    def __repr__(self):
        return "<x:%s y:%s v:%s>" % (self.x, self.y, self.variance)
    def __str__(self):
        return "member of Test"

# Mean squared error
def mean_squared_error(expected, predicted):
	return np.sum((expected - predicted) ** 2)/len(expected)

# Squared error
def squared_error(expected, predicted):
	return np.sum((expected - predicted) ** 2)
	
# Absolute residual error
def absolute_residual_error(expected, predicted):
	return np.sum(abs(expected - predicted))/len(expected)
	
# We use Gaussian RBF's with the following transfer function
def transfer_function(x, position, variance):
	return (math.exp((-(x - position)**2) / (2*(variance**2))))

# We use Gaussian RBF's with the following transfer function
def transfer_function_2d(x, y, position_x, position_y, variance):
	return (math.exp((-(euclidean_distance(x, y, position_x, position_y))**2) / (2*(variance**2))))

def euclidean_distance(x1, y1, x2, y2):
	return (math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2)))
	
def sin_function(x):
	return math.sin(2*x)
	
def square_function(x):
	if (math.sin(2*x) > 0): 
		return 1
	return -1
	
# Sets values in input_pattern that are >= 0 to 1 and values < 0 to -1
def binary(input_pattern):
	for v in range (0, len(input_pattern)):
		if (input_pattern[v] >= 0):
			input_pattern[v] = 1
		else:
			input_pattern[v] = -1
	return input_pattern
	
# Adds noise to input_pattern and then returns it as output_pattern
def noise(input_pattern):
	output_pattern = []
	for i in range(0, len(input_pattern)):
		output_pattern.append(input_pattern[i] + np.random.normal(0, 0.1, 1)[0])
	return output_pattern

# Adds shuffles pattern A and B
def shuffle(A, B):
	temp_A = np.copy(A)
	temp_B = np.copy(B)
	C = np.asarray(np.arange(0, len(A), 1))
	random.shuffle(C)
	for i in range(0, len(C)):
		A[i] = temp_A[C[i]]
		B[i] = temp_B[C[i]]

# Adds shuffles pattern A and B
def shuffle_3(A, B, C):
	temp_A = np.copy(A)
	temp_B = np.copy(B)
	temp_C = np.copy(C)
	D = np.asarray(np.arange(0, len(A), 1))
	random.shuffle(D)
	for i in range(0, len(D)):
		A[i] = temp_A[D[i]]
		B[i] = temp_B[D[i]]
		C[i] = temp_C[D[i]]

simulation_results = []
random_simulation_results = []
Batch_error = []
Batch_time = []
Two_layer_error = []
Two_layer_time = []
sum_first_milestone = 0
sum_second_milestone = 0
sum_third_milestone = 0
for runs in range(1, 26):#11):
	print("Epochs: ", runs)
	nodes = 50
	learning_rate = 0.1
	#learning_rate = float(0.1) / (2**learning)
	#print(learning_rate)
	# Generate function data

	# SIN DATA----------------------------------------------------------------------------------------------------------
	# Training patterns - Generate values between 0 and 2pi with step length 0.1 using our sin_function
	sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
	#sin_training_input_pattern = noise(sin_training_input_pattern)
	sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))
	# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our sin_function
	sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	#sin_test_input_pattern = noise(sin_test_input_pattern)
	sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))
	# SIN DATA----------------------------------------------------------------------------------------------------------

	# SQUARE DATA-------------------------------------------------------------------------------------------------------
	# Training patterns - Generate values between 0 and 2pi with step length 0.1 using our square_function
	square_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
	square_training_output_pattern = list(map(square_function, square_training_input_pattern))

	# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our square_function
	square_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	square_test_output_pattern = list(map(square_function, square_test_input_pattern))
	# SQUARE DATA-------------------------------------------------------------------------------------------------------
	
	errors = []
	RANDOM_errors = []
	#for nodes in range(0, 101):
	# Initiate RBF nodes and WEIGHTS
	start = time.time()
	NUM_NODES_ROW = nodes # Using len(sin_training_output_pattern) or len(square_training_output_pattern) gives good results
	NUM_NODES_COL =	1
	variance = 0.1
	mu, sigma = 0, 0.1 # used for weight initialization
	RBF_Nodes = []
	weight = []

	RANDOM_RBF_Nodes = []
	for c in range(0, NUM_NODES_COL):
		for r in range(0, NUM_NODES_ROW):
			x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW/2)) + r * ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW)
			#y = (y_lower_interval + ((y_upper_interval - y_lower_interval)/NUM_NODES_COL/2)) + c * ((y_upper_interval - y_lower_interval)/NUM_NODES_COL)
			y = 0
			RANDOM_x = float(random.randint(x_lower_interval*1000, math.ceil(x_upper_interval)*1000))/1000
			RANDOM_y = 0
			#print(float(random.randint(x_lower_interval*1000, math.ceil(x_upper_interval)*1000))
			#print(x, y)
			weight.append(np.random.normal(mu, sigma, 1)[0])
			RBF_Nodes.append(Node(x, y, variance))
			RANDOM_RBF_Nodes.append(Node(RANDOM_x, RANDOM_y, variance))

	# Calculate SIN phi
	sin_train_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	sin_test_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	RANDOM_sin_train_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	RANDOM_sin_test_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	for p in range (0, len(sin_training_input_pattern)):
		for n in range (0, len(RBF_Nodes)):
			sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			#sin_train_phi[p][n] = transfer_function_2d(sin_training_input_pattern[p], sin_training_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			#sin_test_phi[p][n] = transfer_function_2d(sin_test_input_pattern[p], sin_test_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			RANDOM_sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], RANDOM_RBF_Nodes[n].x, RANDOM_RBF_Nodes[n].variance)
			RANDOM_sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], RANDOM_RBF_Nodes[n].x, RANDOM_RBF_Nodes[n].variance)
	
	'''
	# Calculate SQUARE phi
	square_train_phi = np.zeros((len(square_training_input_pattern), len(RBF_Nodes)))
	square_test_phi = np.zeros((len(square_training_input_pattern), len(RBF_Nodes)))
	for p in range (0, len(square_training_input_pattern)):
		for n in range (0, len(RBF_Nodes)):
			square_train_phi[p][n] = transfer_function(square_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			square_test_phi[p][n] = transfer_function(square_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			#square_train_phi[p][n] = transfer_function_2d(square_training_input_pattern[p], square_training_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			#square_test_phi[p][n] = transfer_function_2d(square_test_input_pattern[p], square_test_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
	'''
	'''
	# Least squares
	# SIN calculate weights and absolute residual error 
	sin_least_squares_weight = np.linalg.solve(np.matmul(sin_train_phi.T, sin_train_phi), np.matmul(sin_train_phi.T, sin_training_output_pattern))
	sin_least_squares_output_pattern = np.sum(sin_test_phi * sin_least_squares_weight, axis = 1)
	Batch_error.append(absolute_residual_error(sin_test_output_pattern, sin_least_squares_output_pattern))
	print("Nodes:", nodes, "SIN Least squares absolute residual error:", absolute_residual_error(sin_test_output_pattern, sin_least_squares_output_pattern))
	'''
	'''
	#SQUARE calculate weights and absolute residual error
	square_least_squares_weight = np.linalg.solve(np.matmul(square_train_phi.T, square_train_phi), np.matmul(square_train_phi.T, square_training_output_pattern))
	square_least_squares_output_pattern = np.sum(square_test_phi * square_least_squares_weight, axis = 1)
	binary(square_least_squares_output_pattern)
	print("Nodes:", nodes, "SQUARE Least squares absolute residual error:", absolute_residual_error(square_test_output_pattern, square_least_squares_output_pattern))
	'''
	
	#end = time.time()
	#Batch_time.append(end - start)
	'''
	print("Batch execution time: ", end - start)
	if (run == 1):
		ax = plt.gca()
		ax.plot(sin_test_input_pattern, sin_test_output_pattern)
		ax.plot(sin_test_input_pattern, sin_least_squares_output_pattern)
	'''
	
	# Delta rule
	# Initiate weights
	sin_sequential_weight  = []
	RANDOM_sin_sequential_weight  = []
	#square_sequential_weight = []
	#sin_batch_weight = []
	#square_batch_weight = []
	for i in range(0, len(weight)):
		sin_sequential_weight.append(weight[i])
		RANDOM_sin_sequential_weight.append(weight[i])
		#square_sequential_weight.append(weight[i])
		#sin_batch_weight.append(weight[i])
		#square_batch_weight.append(weight[i])

	epochs = 1000
	first_milestone = False
	second_milestone = False
	third_milestone = False

	# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
	# SIN
	for i in range(0, epochs):
		shuffle_3(sin_training_output_pattern, sin_train_phi, RANDOM_sin_train_phi)
		for o in range(0, len(sin_training_output_pattern)):
			sin_sequential_weight = sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(sin_train_phi[o] * sin_sequential_weight))*(sin_train_phi[o]))
			RANDOM_sin_sequential_weight = RANDOM_sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(RANDOM_sin_train_phi[o] * RANDOM_sin_sequential_weight))*(RANDOM_sin_train_phi[o]))
		sin_sequential_output_pattern = np.sum(sin_test_phi * sin_sequential_weight, axis = 1)
		RANDOM_sin_sequential_output_pattern = np.sum(RANDOM_sin_test_phi * RANDOM_sin_sequential_weight, axis = 1)
		errors.append(absolute_residual_error(sin_test_output_pattern, sin_sequential_output_pattern))
		RANDOM_errors.append(absolute_residual_error(sin_test_output_pattern, RANDOM_sin_sequential_output_pattern))
		err = absolute_residual_error(sin_test_output_pattern, sin_sequential_output_pattern)
		#print("Epoch:", i, "SIN Sequential Delta rule error:", err)
	simulation_results.append(errors)
	random_simulation_results.append(RANDOM_errors)
	'''	
		if(err < 0.1):
			if(first_milestone == False):
				print("First milestone", i)
				first_milestone = True
				sum_first_milestone = sum_first_milestone + i
				print("First milestone sum: ", sum_first_milestone, "runs :", runs)
		if(err < 0.01):
			if(second_milestone == False):
				print("Second milestone", i)
				second_milestone = True
				sum_second_milestone = sum_second_milestone + i
				print("Second milestone sum: ", sum_second_milestone, "runs :", runs)
		if(err < 0.001):
			if(third_milestone == False):
				print("Third milestone", i)
				third_milestone = True
				sum_third_milestone = sum_third_milestone + i
				print("Third milestone sum: ", sum_third_milestone, "runs :", runs)
				i = 10000000
			
	
	if(third_milestone == False):
		print("Third milestone failed to find solution")
	print("")
	'''
	
		#print("Epoch:", i, "RANDOM SIN Sequential Delta rule error:", absolute_residual_error(sin_test_output_pattern, RANDOM_sin_sequential_output_pattern))
	#print("Nodes:", nodes, "SIN Sequential Delta rule error:", squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
	
	'''
	
	# SQUARE
	for i in range(0, epochs):
		for o in range(0, len(square_training_output_pattern)):
			square_sequential_weight = square_sequential_weight + (learning_rate*(square_training_output_pattern[o] - np.sum(square_train_phi[o] * square_sequential_weight))*(square_train_phi[o]))
		square_sequential_output_pattern = np.sum(square_test_phi * square_sequential_weight, axis = 1)
		binary(square_sequential_output_pattern)
		#print("Epoch:", i, "SQUARE Sequential Delta rule error:", squared_error(square_test_output_pattern, square_sequential_output_pattern))
	binary(square_sequential_output_pattern)
	print("Nodes:", nodes, "SQUARE Sequential Delta rule error:", squared_error(square_test_output_pattern, square_sequential_output_pattern))
	# Sequential Delta rule--------------------square_least_squares_output_pattern------------------------------------------------------------------------------------------------------------------------
	
	# Batch Delta rule-------------------------------------------------------------------------------------------------------------------------------------------------
	# SIN
	for i in range(0, epochs):
		sin_batch_output_pattern = np.sum(sin_train_phi * sin_batch_weight, axis = 1)
		sin_batch_weight = sin_batch_weight + (learning_rate*np.sum((sin_training_output_pattern - sin_batch_output_pattern)*sin_train_phi.T, axis = 1))
		#print("Epoch:", i, "SIN Batch Delta rule error:", squared_error(sin_test_output_pattern, sin_batch_output_pattern))
	sin_batch_output_pattern = np.sum(sin_test_phi * sin_batch_weight, axis = 1)
	print("Nodes:", nodes, "SIN Batch Delta rule error:", squared_error(sin_test_output_pattern, sin_batch_output_pattern))

	# SQUARE
	for i in range(0, epochs):
		square_batch_output_pattern = np.sum(square_train_phi * square_batch_weight, axis = 1)
		square_batch_weight = square_batch_weight + (learning_rate*np.sum((square_training_output_pattern - square_batch_output_pattern)*square_train_phi.T, axis = 1))
		binary(square_batch_output_pattern)
		#print("Epoch:", i, "SQUARE Batch Delta rule error:", squared_error(square_test_output_pattern, square_batch_output_pattern))
	square_batch_output_pattern = np.sum(square_test_phi * square_batch_weight, axis = 1)
	binary(square_batch_ouabsolute_residual_error(sin_test_output_pattern, RANDOM_sin_sequential_output_pattern)tput_pattern)
	print("Nodes:", nodes, "SQUARE Batch Delta rule error:", squared_error(square_test_output_pattern, square_batch_output_pattern))
	# Batch Delta rule-------------------------------------------------------------------------------------------------------------------------------------------------
	'''
	'''
	start = time.time()
	#ANN
	parser = argparse.ArgumentParser(description='MLP network for Mackey-Glass time series predictions.')
	parser.add_argument('-n', '--hidden-nodes', type=int, nargs='+', default=5,
		           help='number of nodes in the hidden layers (max 8 per layer)')
	parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
		           help='the learning rate, controls how fast it converges')
	parser.add_argument('-a', '--alpha', type=float, default=0.0001,
		           help='the L2 regularization factor')
	args = parser.parse_args()

	np.set_printoptions(threshold=np.nan) #Always print the whole matrix

	training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
	training_target_pattern = list(map(sin_function, training_input_pattern))
	training_input_pattern = training_input_pattern.reshape(1, -1)
	test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	test_target_pattern = list(map(sin_function, test_input_pattern))

	reg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=args.hidden_nodes, early_stopping=True, max_iter=100000,
		           learning_rate_init=args.learning_rate, alpha=args.alpha, batch_size=len(training_input_pattern))
	reg = reg.fit(np.transpose(training_input_pattern), training_target_pattern)
	output = reg.predict(np.transpose(training_input_pattern))

	print("Two-layer absolute residual error:", absolute_residual_error(sin_test_output_pattern, output))
	Two_layer_error.append(absolute_residual_error(sin_test_output_pattern, output))
	
	end = time.time()
	print("Two-layer execution time: ", end - start)
	Two_layer_time.append(end - start)
	'''
	
#print("Two-layer average absolute residual error: ", sum(Two_layer_error)/10)
'''
print("First milestone average", sum_first_milestone/runs)
print("Second milestone average: ", sum_second_milestone/runs)
print("Third milestone average: ", sum_third_milestone/runs)
#print("Third milestone average: ", runs)
'''
'''
sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))
sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))
print("FAODSOF", runs)
print("")
print("Batch average absolute residual error: ", sum(Batch_error)/runs)
print("Batch average execution time: ", sum(Batch_time)/runs)
print("Two-layer average absolute residual error: ", sum(Two_layer_error)/runs)
print("Two-layer average execution time: ", sum(Two_layer_time)/runs)
'''
# Plot
#ax = plt.gca()
'''
# Plot nodes
X = []
Y = []
Circles = []
for node in RBF_Nodes:
	X.append(node.x)
	Y.append(node.y)
	Circles.append(plt.Circle((node.x, node.y), node.variance, color='k', fill=False))

ax.plot(X, Y, "ro")
for circle in Circles:
	ax.add_artist(circle)
'''
'''
# Plot data
ax.plot(sin_test_input_pattern, sin_test_output_pattern)
ax.plot(sin_test_input_pattern, sin_sequential_output_pattern)
ax.plot(sin_test_input_pattern, sin_least_squares_output_pattern)
#ax.plot(square_test_input_pattern, square_test_output_pattern)
#ax.plot(square_test_input_pattern, square_least_squares_output_pattern)
#ax.plot(sin_test_input_pattern, output)
plt.ylim( (-3, 3) )
ax.legend(['y = sin(2x)', 'y = Least squares', 'y = Delta'])
plt.ylabel('Estimated function value')
plt.xlabel('Input value')
#plt.ylim( (0, 0.5) )
#print(simulation_results[0][0:epochs])
#print(simulation_results[3][0:epochs])
'''
'''
#ax.plot(simulation_results[0][0:1000])
ax.plot(simulation_results[0])
ax.plot(simulation_results[1])
ax.plot(simulation_results[2])
ax.plot(simulation_results[3])

ax.legend(['Learning rate = 0.1', 'Learning rate = 0.05', 'Learning rate = 0.025', 'Learning rate = 0.0125'])
plt.ylabel('Absolute residual error')
plt.xlabel('Epochs')
'''
ax = plt.gca()
'''
ax.plot(simulation_results[0], color='C0')
ax.plot(simulation_results[1], color='C0')
ax.plot(simulation_results[2], color='C0')
ax.plot(simulation_results[3], color='C0')
ax.plot(simulation_results[4], color='C0')
ax.plot(simulation_results[5], color='C0')
ax.plot(simulation_results[6], color='C0')
ax.plot(simulation_results[7], color='C0')
ax.plot(simulation_results[8], color='C0')
ax.plot(simulation_results[9], color='C0')
ax.plot(random_simulation_results[0], color='C3')
ax.plot(random_simulation_results[1], color='C3')
ax.plot(random_simulation_results[2], color='C3')
ax.plot(random_simulation_results[3], color='C3')
ax.plot(random_simulation_results[4], color='C3')
ax.plot(random_simulation_results[5], color='C3')
ax.plot(random_simulation_results[6], color='C3')
ax.plot(random_simulation_results[7], color='C3')
ax.plot(random_simulation_results[8], color='C3')
ax.plot(random_simulation_results[9], color='C3')
'''


a = np.zeros(len(simulation_results[0]))
b = np.zeros(len(simulation_results[0]))

for i in range(0, len(a)):
	for j in range(0, len(simulation_results)):
		a[i] = a[i] + simulation_results[j][i]
		b[i] = b[i] + random_simulation_results[j][i]
	a[i] = a[i]/10
	b[i] = b[i]/10

#print(a)
#print(b)

ax.plot(a)
ax.plot(b)

ax.legend(['y = Manually placed', 'y = Randomly placed'])
plt.ylabel('Absolute residual error')
plt.xlabel('Epochs')

plt.show()


