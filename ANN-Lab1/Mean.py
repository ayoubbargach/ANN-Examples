import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

NUM_POINTS = 200
learning_rate = 0.0001

# Class that represents a 2d input and what type it should be classified as
class Point:
	def __init__(self, x, y, c):
		self.x = x
		self.y = y
		self.classification = c

# Mean squared error
def mse( expected, predicted ):
	return np.sum((expected - predicted) ** 2)/len(expected)

# Check amount of incorrect classifications
def incorrect_classifications( output, target ):
	incorrect_classifications = 0
	for i in range(0, len(output)):
		if(output[i] != target[i]):
			incorrect_classifications += 1
	return incorrect_classifications

def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

all_learning_curves = []
all_classification_curves = []
test_all_learning_curves = []
test_all_classification_curves = []
for t in range(0, 100):
	# Generate points with multivariate normal distribution

	# LINEARLY-SEPARABLE DATA
	#meanA = [5, 3]
	#meanB = [-5, 3]
	#covA = [[1, 0], [-5, 10]]
	#covB = [[1, 0], [-5, 10]]

	# NOT LINEARLY SEPERABLE DATA
	meanA = [5, 3]
	meanB = [-5, 3]
	covA = [[9, 0], [-5, 10]]
	covB = [[9, 0], [-5, 10]]

	points = []

	Class_A = np.random.multivariate_normal(meanA, covA, NUM_POINTS//2).T
	for x in range(0, NUM_POINTS//2):
	    points.append(Point(Class_A[0][x], Class_A[1][x], 1))

	Class_B = np.random.multivariate_normal(meanB, covB, NUM_POINTS//2).T
	for x in range(0, NUM_POINTS//2):
	    points.append(Point(Class_B[0][x], Class_B[1][x], -1))

	shuffle(points)

	# Initialize input output and target pattern based on earlier generated points
	input_pattern = []
	output_pattern = []
	target_pattern = []
	for point in points:
	    input_pattern.append([point.x, point.y])
	    target_pattern.append(point.classification)

	input_pattern = np.transpose(input_pattern)
	pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)


	# TEST INITIATION
	test_points = []

	Class_A = np.random.multivariate_normal(meanA, covA, 50).T
	for x in range(0, 50):
	    test_points.append(Point(Class_A[0][x], Class_A[1][x], 1))

	Class_B = np.random.multivariate_normal(meanB, covB, 50).T
	for x in range(0, 50):
	    test_points.append(Point(Class_B[0][x], Class_B[1][x], -1))

	shuffle(test_points)

	# Initialize input output and target pattern based on earlier generated points
	test_input_pattern = []
	test_target_pattern = []
	for point in test_points:
	    test_input_pattern.append([point.x, point.y])
	    test_target_pattern.append(point.classification)

	test_input_pattern = np.transpose(test_input_pattern)
	#test_pat = np.append(test_input_pattern, [[1] * NUM_POINTS], 0)


	# Initiate weights using small random numbers drawn from the normal distribution with zero mean
	weight = []
	mu, sigma = 0, 0.1
	weight.append(np.random.normal(mu, sigma, 1)[0])
	weight.append(np.random.normal(mu, sigma, 1)[0])
	weight.append(np.random.normal(mu, sigma, 1)[0])

	def normalize( v ):
		result = 0
		for x in v :
			result = result + x*x

		result = math.sqrt( result )

		for x in range(0, len(v)):
			v[x] = v[x] / result
		return v


	mean_squared_error = 1
	previous_squared_error = 2
	learning_curve = []
	classification_curve = []
	test_learning_curve = []
	test_classification_curve = []

	# Perceptron Learning
	# Sequential
	'''
	for x in range(0, 20):
		output_pattern = []
		for y in range(0, NUM_POINTS):
			output_pattern.append(np.dot(weight, np.transpose(pat)[y]))
			if output_pattern[y] > 0:
				output_pattern[y] = 1
			else:
				output_pattern[y] = -1
			error = target_pattern[y] - output_pattern[y]
			weight += learning_rate * error * np.transpose(pat)[y]
		mean_squared_error = mse(np.array(target_pattern), np.array(output_pattern))
		learning_curve.append(mean_squared_error)
	'''

	'''
	# Batch
	error = []
	for x in range(0, 20):
		output_pattern = np.dot(np.transpose(weight), pat)
		for y in range(0, 200):
			if output_pattern[y] > 0:
				output_pattern[y] = 1
			else:
				output_pattern[y] = -1
		error = np.subtract(target_pattern, output_pattern)
		weight += np.dot(learning_rate, np.dot(error, np.transpose(pat)))
		mean_squared_error = mse(np.array(target_pattern), np.array(output_pattern))
		learning_curve.append(mean_squared_error)
		#print(np.sum(error))
	
	'''

	'''
	# Delta Learning
	error = []
	# Sequential
	output_patterns = []
	for x in range(0, 20):
		output_pattern = []
		for y in range(0, NUM_POINTS):
			output_pattern.append(np.dot(weight, np.transpose(pat)[y]))
			#output_patterns.append(output_pattern)
			e = target_pattern[y] - output_pattern[y]
			#error.append(e)
			weight += learning_rate * e * np.transpose(pat)[y]
		mean_squared_error = mse(np.array(target_pattern), np.array(output_pattern))
		learning_curve.append(mean_squared_error)
		#print(error)
	'''

	'''     
	# Batch
	for x in range(0, 20):
		output_pattern = np.dot(np.transpose(weight), pat)
		error = np.subtract(target_pattern, output_pattern)
		mean_squared_error = mse(target_pattern, output_pattern)
		learning_curve.append(mean_squared_error)
		for y in range(0, 200):
			if output_pattern[y] > 0:
				output_pattern[y] = 1
			else:
				output_pattern[y] = -1
		classification_curve.append(incorrect_classifications(output_pattern, target_pattern))
		
		weight += np.dot(learning_rate, np.dot(error, np.transpose(pat)))
		#normalize(weight)
		#print(mean_squared_error)

	'''

	# Two-layer perceptron
	STEP_LENGTH = 2
	input_size = 2
	output_size = 1
	hidden_layer_size = 8

	# Initiate weights using small random numbers drawn from the normal distribution with zero mean
	mu, sigma = 0, 0.1

	v = []
	dv = []
	for x in range(0, hidden_layer_size):
		v.append(np.random.normal(mu, sigma, input_size + 1))
		dv.append(np.random.normal(mu, sigma, input_size + 1))

	v = np.array(v)
	dv = np.array(dv)

	w = []
	dw = []
	for x in range(0, output_size):
		w.append(np.random.normal(mu, sigma, hidden_layer_size + 1))
		dw.append(np.random.normal(mu, sigma, hidden_layer_size + 1))

	w = np.array(w)
	dw = np.array(dw)
	alpha = 0.9

	pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)

	batch_size = NUM_POINTS
	# batch_size = NUM_POINTS # BATCH
	# batch_size = 1 # SEQUENTIAL
	# batch_size = NUM_POINTS // x # MINIBATCH
	for x in range(0, 200):
		total_output = np.array([])
		for i in range (0, math.ceil(NUM_POINTS / batch_size)):
			batch_input = np.array(input_pattern)[:, i * batch_size : (i * batch_size) + batch_size]
			batch_target = target_pattern[i * batch_size : (i * batch_size) + batch_size]
			batch_pat = np.append(batch_input, [[1] * batch_size], 0)
			# Forward pass
			h_input = np.dot(v, np.append(batch_input, [[1] * batch_size], 0))
			h_output = np.append(phi( h_input ), [[1] * batch_size], 0)
			o_input = np.dot(w, h_output)
			o_output = phi( o_input )
			total_output = np.append(total_output, o_output)

			# Backward pass
			delta_o = (o_output - np.asarray(batch_target)) * ((1 + o_output) * (1 - o_output)) * 0.5
			delta_h = (np.dot(np.transpose(w), delta_o)) * ((1 + h_output) * (1 - h_output)) * 0.5
			delta_h = np.delete(delta_h, -1, 0)
		
			# Weight update
			dv = (dv * alpha) - (np.dot(delta_h, np.transpose(batch_pat))) * (1 - alpha)
			dw = (dw * alpha) - (np.dot(delta_o, np.transpose(h_output))) * (1 - alpha)
			v = v + dv * learning_rate
			w = w + dw * learning_rate

		#print(o_output)
		#print(np.array(total_output).shape)
		#print(total_output)
		mean_squared_error = mse(np.array(target_pattern), total_output)
		learning_curve.append(mean_squared_error)
		for y in range(0, NUM_POINTS):
			if total_output[y] > 0:
				total_output[y] = 1
			else:
				total_output[y] = -1
		classification_curve.append(incorrect_classifications(total_output, target_pattern))


		# error = np.subtract(target_pattern, o_output)

		#print(np.sum(error))

		# Forward pass
		h_input = np.dot(v, np.append(test_input_pattern, [[1] * 100], 0))
		h_output = np.append(phi( h_input ), [[1] * 100], 0)
		o_input = np.dot(w, h_output)
		o_output = phi( o_input )

		mean_squared_error = mse(np.array(test_target_pattern), np.array(o_output))
		test_learning_curve.append(mean_squared_error)
		for y in range(0, 100):
			if o_output[0][y] > 0:
				o_output[0][y] = 1
			else:
				o_output[0][y] = -1
		test_classification_curve.append(incorrect_classifications(o_output[0], test_target_pattern))

		#print(incorrect_classifications(o_output[0], test_target_pattern))

	#print(classification_curve)
	all_learning_curves.append(learning_curve)
	all_classification_curves.append(classification_curve)
	test_all_learning_curves.append(test_learning_curve)
	test_all_classification_curves.append(test_classification_curve)


mean_learning_curve = np.mean(all_learning_curves, axis=0)
std_learning_curve = np.std(all_learning_curves, axis=0)
mean_classification_curve = np.divide(np.mean(all_classification_curves, axis=0), NUM_POINTS)

test_mean_learning_curve = np.mean(test_all_learning_curves, axis=0)
test_std_learning_curve = np.std(test_all_learning_curves, axis=0)
test_mean_classification_curve = np.divide(np.mean(test_all_classification_curves, axis=0), 100)

'''
# Plot the points
positive = []
negative = []
for x in range(0, NUM_POINTS):
    if target_pattern[x] == 1:
	    positive.append([input_pattern[0][x], input_pattern[1][x]])
    else:
	    negative.append([input_pattern[0][x], input_pattern[1][x]])

positive = np.transpose(positive)
negative = np.transpose(negative)

plt.plot(positive[0], positive[1], 'x')
plt.plot(negative[0], negative[1], 'o')

# Plot separating line
y1 = - (weight[0] / weight[1]) * -5 - (weight[2] / weight[1])
y2 = - (weight[0] / weight[1]) * 5 - (weight[2] / weight[1])

plt.plot([-5, 5], [y1, y2])
plt.axis('equal')
plt.show()
'''

plt.fill_between(range(0, len(std_learning_curve)), mean_learning_curve - std_learning_curve, mean_learning_curve + std_learning_curve, alpha=0.1, color="b")
plt.plot(mean_learning_curve)
#plt.plot(mean_classification_curve)
#plt.plot(test_mean_classification_curve)
#plt.legend(['y = Training set', 'y = Test set'])
plt.show()






