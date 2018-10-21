import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

def CL(RBF_Nodes, input_pattern, leaky = False):
  iters = 1000
  cl_learning_rate = 0.1
  for _ in range (0, iters):

    # pick random training vector
    i = random.randint(0, len(input_pattern) - 1)
    training_vector = np.asarray((input_pattern[i]))

    # find closest rbf_node
    closest_node = None
    closest_distance = float('inf')
    for node in RBF_Nodes:
      npNode = np.asarray((node.x))
      distance = np.linalg.norm(training_vector - npNode)
      if distance < closest_distance:
        closest_distance = distance
        closest_node = node

    if closest_node == None:
      continue

    # move closest rbf_node closer to traning vector, dw = eta(x - w)
    delta_node = cl_learning_rate * (training_vector - np.asarray((closest_node.x)))

    closest_node.x += delta_node

    # consider strategy for dead units (e.g., leaky cl)
    if leaky:
      leaky_learning_rate = 0.01
      for node in RBF_Nodes:
        if node != closest_node:
          # use gauss function to limit how leaky it is, nodes further away are less affected
          npNode = np.asarray((node.x))
          distance = np.linalg.norm(training_vector - npNode)
          gauss_factor = transfer_function(distance, 0, 0.5)
          delta_node = gauss_factor * leaky_learning_rate * (training_vector - np.asarray((node.x)))
          node.x += delta_node

def adjust_widths(RBF_Nodes):
  for i in range(0, len(RBF_Nodes)):
    max_dist = 0
    current_node = RBF_Nodes[i]

    if i > 0:
      prev_node = RBF_Nodes[i - 1]
      dist = current_node.x - prev_node.x
      if dist > max_dist:
        max_dist = dist

    if i < len(RBF_Nodes) - 1:
      next_node = RBF_Nodes[i + 1]
      dist = next_node.x - current_node.x
      if dist > max_dist:
        max_dist = dist

    current_node.variance = max_dist * 0.65

x_lower_interval = 0
x_upper_interval = 2*math.pi
y_lower_interval = -2
y_upper_interval = 2
step_length = 0.1

learning_rate = 0.01

random.seed(a=None)

# Class that represents a 2d input and what type it should be classified as
class Node:
    def __init__(self, x, v):
        self.x = x
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


nodes = 50
# Generate function data

# SIN DATA----------------------------------------------------------------------------------------------------------
# Training patterns - Generate values between 0 and 2pi with step length 0.1 using our sin_function
sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
#sin_training_input_pattern[0] += 0.0001
sin_training_input_pattern = noise(sin_training_input_pattern)
sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))

# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our sin_function
sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
sin_test_input_pattern = noise(sin_test_input_pattern)
sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))

# SIN DATA----------------------------------------------------------------------------------------------------------

errors = defaultdict(list)
RANDOM_errors = []
#for nodes in range(0, 101):
# Initiate RBF nodes and WEIGHTS
variance = 0.1
mu, sigma = 0, 0.1 # used for weight initialization
RBF_Nodes = []
comp_RBF_Nodes = []
weight = []
comp_weight = []

for c in range(0, nodes):
  x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/nodes/2)) + c * ((x_upper_interval - x_lower_interval)/nodes)
  weight.append(np.random.normal(mu, sigma, 1)[0])
  comp_weight.append(np.random.normal(mu, sigma, 1)[0])
  RBF_Nodes.append(Node(x, variance))
  comp_RBF_Nodes.append(Node(x, variance))

CL(comp_RBF_Nodes, sin_training_input_pattern, False)
adjust_widths(comp_RBF_Nodes)

# Calculate SIN phi
sin_train_phi = np.zeros((len(sin_training_input_pattern), nodes))
comp_sin_train_phi = np.zeros((len(sin_training_input_pattern), nodes))
sin_test_phi = np.zeros((len(sin_training_input_pattern), nodes))
comp_sin_test_phi = np.zeros((len(sin_training_input_pattern), nodes))
for p in range (0, len(sin_training_input_pattern)):
  for n in range (0, nodes):
    sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
    comp_sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], comp_RBF_Nodes[n].x, comp_RBF_Nodes[n].variance)
    sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
    comp_sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], comp_RBF_Nodes[n].x, comp_RBF_Nodes[n].variance)

# Delta rule
# Initiate weights
sin_sequential_weight  = []
comp_sin_sequential_weight  = []
for i in range(0, len(comp_weight)):
  sin_sequential_weight.append(weight[i])
  comp_sin_sequential_weight.append(comp_weight[i])

epochs = 1000

# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
# SIN
for i in range(0, epochs):
  shuffle_3(sin_training_output_pattern, sin_train_phi, comp_sin_train_phi)
  for o in range(0, len(sin_training_output_pattern)):
    sin_sequential_weight = sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(sin_train_phi[o] * sin_sequential_weight))*(sin_train_phi[o]))
    comp_sin_sequential_weight = comp_sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(comp_sin_train_phi[o] * comp_sin_sequential_weight))*(comp_sin_train_phi[o]))
  sin_sequential_output_pattern = np.sum(sin_test_phi * sin_sequential_weight, axis = 1)
  comp_sin_sequential_output_pattern = np.sum(comp_sin_test_phi * comp_sin_sequential_weight, axis = 1)
  errors['normal'].append(absolute_residual_error(sin_test_output_pattern, sin_sequential_output_pattern))
  errors['comp'].append(absolute_residual_error(sin_test_output_pattern, comp_sin_sequential_output_pattern))
  print("Epoch:", i, "SIN Sequential Delta rule error:", absolute_residual_error(sin_test_output_pattern, comp_sin_sequential_output_pattern))

# Plot
ax = plt.gca()

# Plot nodes
X = []
Y = []
Circles = []
for node in RBF_Nodes:
  X.append(node.x)
  Y.append(0)
  Circles.append(plt.Circle((node.x, 0), node.variance, color='k', fill=False))

comp_X = []
comp_Y = []
comp_Circles = []
for node in comp_RBF_Nodes:
  comp_X.append(node.x)
  comp_Y.append(0)
  comp_Circles.append(plt.Circle((node.x, 0), node.variance, color='k', fill=False))

plot_error = True
if plot_error:
  ax.plot(errors['normal'])
  ax.plot(errors['comp'])
  ax.legend(['Manually placed', 'Competitive learning'])
  plt.ylabel('Absolute residual error')
  plt.xlabel('Epochs')
  plt.xlim([0, 200])
else:
  comp = True
  if comp:
    ax.plot(X, Y, "ro")
    for circle in comp_Circles:
      ax.add_artist(circle)
    ax.plot(sin_test_input_pattern, comp_sin_sequential_output_pattern)
    ax.plot(sin_test_input_pattern, sin_test_output_pattern)
    ax.scatter(sin_test_input_pattern, comp_sin_sequential_output_pattern)
    ax.legend(['Competitive RBF Nodes', 'approximation plot', 'sin2x', 'appromixation scatter'])
    plt.xlim([0, 2*math.pi])
  else:
    ax.plot(X, Y, "ro")
    for circle in Circles:
      ax.add_artist(circle)
    ax.plot(sin_test_input_pattern, sin_sequential_output_pattern)
    ax.plot(sin_test_input_pattern, sin_test_output_pattern)
    ax.scatter(sin_test_input_pattern, sin_sequential_output_pattern)
    ax.legend(['RBF Nodes', 'approximation plot', 'sin2x', 'appromixation scatter'])
    plt.xlim([0, 2*math.pi])

plt.show()



