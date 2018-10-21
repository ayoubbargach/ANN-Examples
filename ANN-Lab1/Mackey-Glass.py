import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import shuffle

def normalize( v ):
   result = 0
   for x in v :
      result = result + x*x

   result = math.sqrt( result )

   for x in range(0, len(v)):
      v[x] = v[x] / result
   return v

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

POINTS_TO_GEN = 1500
OFFSET = 300
NUM_POINTS = POINTS_TO_GEN - OFFSET
learning_rate = 0.0001

TRAINING_SIZE = 800
VALIDATION_SIZE = 200
TEST_SIZE = 200

def MGE(t, x):
	return x[t] + ((0.2 * x[t-25]) / (1 + x[t-25]**10)) - 0.1 * x[t]


prev_x = []
# Add 25 zero values so that prev_x[t] will return 0 if t is lower than 0
for t in range(0, 25):
	prev_x.append(0)

x = 1.5
# Calculate x values
prev_x.append(x)
for t in range(25, POINTS_TO_GEN + 50):
	prev_x.append(MGE(t, prev_x))

# Calculate input values for the ANN
input_pattern = []
target_pattern = []
for t in range(OFFSET+50, POINTS_TO_GEN + 50):
	input_pattern.append([prev_x[t-25], prev_x[t-20], prev_x[t-15], prev_x[t-10], prev_x[t-5]])
	target_pattern.append(prev_x[t])

input_pattern = np.array(input_pattern)
print(len(target_pattern))
#input_pattern = (input_pattern - input_pattern.mean(axis=0))/input_pattern.var(axis=0)
input_pattern = np.transpose(input_pattern)
target_pattern = np.array(target_pattern)
target_pattern = (target_pattern - target_pattern.mean(axis=0))/target_pattern.std(axis=0) / 2.5

#for t in range(0, 5):
	#input_pattern[t] = normalize(input_pattern[t])

#target_pattern = normalize(target_pattern)

# Two-layer perceptron
def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

input_size = 5
output_size = 1
hidden_layer_size = 15

# Initiate weights using small random numbers drawn from the normal distribution with zero mean
mu, sigma = 0, 0.1

v = []
dv = []
for i in range(0, hidden_layer_size):
	v.append(np.random.normal(mu, sigma, input_size + 1))
	dv.append(np.random.normal(mu, sigma, input_size + 1))

v = np.array(v)
dv = np.array(dv)

w = []
dw = []
for i in range(0, output_size):
	w.append(np.random.normal(mu, sigma, hidden_layer_size + 1))
	dw.append(np.random.normal(mu, sigma, hidden_layer_size + 1))

w = np.array(w)
dw = np.array(dw)
alpha = 0.9

pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)

o_output = []
for i in range(0, 10000):
	# Forward pass
	h_input = np.dot(v, np.append(input_pattern, [[1] * NUM_POINTS], 0))
	h_output = np.append(phi( h_input ), [[1] * NUM_POINTS], 0)
	o_input = np.dot(w, h_output)
	o_output = phi( o_input )

	# Backward pass
	delta_o = (o_output - np.asarray(target_pattern)) * ((1 + o_output) * (1 - o_output)) * 0.5
	delta_h = (np.dot(np.transpose(w), delta_o)) * ((1 + h_output) * (1 - h_output)) * 0.5
	delta_h = np.delete(delta_h, -1, 0)
	
	# Weight update
	dv = (dv * alpha) - (np.dot(delta_h, np.transpose(pat))) * (1 - alpha)
	dw = (dw * alpha) - (np.dot(delta_o, np.transpose(h_output))) * (1 - alpha)
	v = v + dv * learning_rate
	w = w + dw * learning_rate

	error = np.subtract(target_pattern, o_output)
	print(np.sum(error))

plt.plot(o_output[0])
plt.plot(target_pattern)
plt.legend(['y = ANN', 'y = TARGET'])
plt.ylabel('some numbers')
plt.show()


