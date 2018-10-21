import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

def CL(RBF_Nodes, input_pattern, leaky = False):
  iters = 1000
  cl_learning_rate = 0.25
  for _ in range (0, iters):

    # pick random training vector
    i = random.randint(0, len(input_pattern) - 1)
    training_vector = np.asarray((input_pattern[i][0], input_pattern[i][1]))

    # find closest rbf_node
    closest_node = None
    second_closest_node = None
    closest_distance = float('inf')
    for node in RBF_Nodes:
      npNode = np.asarray((node.x, node.y))
      distance = np.linalg.norm(training_vector - npNode)
      if distance < closest_distance:
        second_closest_node = closest_node
        closest_distance = distance
        closest_node = node

    if closest_node == None:
      continue

    # move closest rbf_node closer to traning vector, dw = eta(x - w)
    delta_node = cl_learning_rate * (training_vector - np.asarray((closest_node.x, closest_node.y)))

    closest_node.x += delta_node[0]
    closest_node.y += delta_node[1]


    # consider strategy for dead units (e.g., leaky cl)
    if leaky:
      leaky_learning_rate = 0.1
      '''
      if second_closest_node == None:
        continue
      #second_winner = RBF_Nodes[random.randint(0, len(RBF_Nodes) - 1)]
      second_winner = second_closest_node
      second_delta_node = leaky_learning_rate * (training_vector - np.asarray((second_winner.x, second_winner.y)))
      second_winner.x += second_delta_node[0]
      second_winner.y += second_delta_node[1]
      '''

      for node in RBF_Nodes:
        if node != closest_node:
          # use gauss function to limit how leaky it is, nodes further away are less affected
          gauss_factor = transfer_function(training_vector, node, 0.05)
          delta_node = gauss_factor * leaky_learning_rate * (training_vector - np.asarray((node.x, node.y)))
          node.x += delta_node[0]
          node.y += delta_node[1]

def adjust_widths(RBF_Nodes):
  for i in range(0, len(RBF_Nodes)):
    min_dist = float('inf')
    current_node = RBF_Nodes[i]
    np_current_node = np.asarray((current_node.x, current_node.y))

    for j in range(0, len(RBF_Nodes)):
      if i == j:
        continue

      np_other_node = np.asarray((RBF_Nodes[j].x, RBF_Nodes[j].y))
      dist = np.linalg.norm(np_current_node - np_other_node)
      if dist < min_dist:
        min_dist = dist

    current_node.variance = max(min_dist * 1.35, 0.2)

def adjust_widths_smart(RBF_Nodes, input_pattern):
  for i in range(0, len(RBF_Nodes)):
    min_dist = float('inf')
    current_node = RBF_Nodes[i]
    np_current_node = np.asarray((current_node.x, current_node.y))

    for j in range(0, len(input_pattern)):
      if i == j:
        continue

      np_other_node = np.asarray((input_pattern[j][0], input_pattern[j][1]))
      dist = np.linalg.norm(np_current_node - np_other_node)
      if dist < min_dist:
        min_dist = dist

    current_node.variance = max(min_dist * 1.35, 0.2)

training_input = []
training_target = []
with open('data_lab2/ballist.dat') as f:
  for line in f:
    line_numbers = [float(x) for x in line.split()]
    training_input.append(np.asarray([line_numbers[0], line_numbers[1]]))
    training_target.append(np.asarray([line_numbers[2], line_numbers[3]]))

test_input = []
test_target = []
with open('data_lab2/balltest.dat') as f:
  for line in f:
    line_numbers = [float(x) for x in line.split()]
    test_input.append(np.asarray([line_numbers[0], line_numbers[1]]))
    test_target.append(np.asarray([line_numbers[2], line_numbers[3]]))
test_target = np.asarray(test_target)

learning_rate = 0.01

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
    def asarray(self):
      return np.asarray([self.x, self.y])

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
  return (math.exp((-(np.linalg.norm(x - position.asarray()))**2) / (2*(variance**2))))

def euclidean_distance(x1, y1, x2, y2):
  return (math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2)))
  
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

# Generate function data

errors = []
comp_errors = []

# Initiate RBF nodes and WEIGHTS
NUM_NODES_ROWS = 5
NUM_NODES_COLS = 5
nodes = NUM_NODES_ROWS * NUM_NODES_COLS
min_x = 0
max_x = 1
min_y = 0
max_y = 0
variance = 0.2
mu, sigma = 0, 0.1 # used for weight initialization
RBF_Nodes = []
comp_RBF_Nodes = []
weight = []
comp_weight = []
for r in range(0, NUM_NODES_ROWS):
  for c in range(0, NUM_NODES_COLS):
    x = c / (NUM_NODES_COLS - 1)
    y = r / (NUM_NODES_ROWS - 1)
    #x = -(NUM_NODES_COLS - 1) * variance + c * 2 * variance + (NUM_NODES_COLS * variance)
    #y = -(NUM_NODES_ROWS - 1) * variance + r * 2 * variance + NUM_NODES_ROWS * variance
    init_weights = np.random.normal(mu, sigma, 2)
    weight.append(np.asarray([init_weights[0], init_weights[1]]))
    comp_weight.append(np.asarray([init_weights[0], init_weights[1]]))
    RBF_Nodes.append(Node(x, y, variance))
    comp_RBF_Nodes.append(Node(x, y, variance))

weight = np.array(weight)
comp_weight = np.array(comp_weight)
CL(comp_RBF_Nodes, training_input, True)
adjust_widths(comp_RBF_Nodes)
#adjust_widths_smart(comp_RBF_Nodes, training_input)

# Calculate SIN phi
training_phi = np.zeros((len(training_input), nodes))
comp_training_phi = np.zeros((len(training_input), nodes))
test_phi = np.zeros((len(training_input), nodes))
comp_test_phi = np.zeros((len(training_input), nodes))
for p in range (0, len(training_input)):
  for n in range (0, len(RBF_Nodes)):
    training_phi[p][n] = transfer_function(training_input[p], RBF_Nodes[n], RBF_Nodes[n].variance)
    comp_training_phi[p][n] = transfer_function(training_input[p], comp_RBF_Nodes[n], comp_RBF_Nodes[n].variance)
    test_phi[p][n] = transfer_function(test_input[p], RBF_Nodes[n], RBF_Nodes[n].variance)
    comp_test_phi[p][n] = transfer_function(test_input[p], comp_RBF_Nodes[n], comp_RBF_Nodes[n].variance)

epochs = 1000
# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
for i in range(0, epochs):
  for o in range(0, len(training_target)):
    weight = weight + np.dot(np.transpose(learning_rate*(training_target[o] - np.dot(training_phi[o].reshape(1, -1), weight))),training_phi[o].reshape(1, -1)).transpose()
    comp_weight = comp_weight + np.dot(np.transpose(learning_rate*(training_target[o] - np.dot(comp_training_phi[o].reshape(1, -1), comp_weight))),comp_training_phi[o].reshape(1, -1)).transpose()
  test_output = np.dot(test_phi, weight)
  comp_test_output = np.dot(comp_test_phi, comp_weight)
  errors.append(absolute_residual_error(test_target, test_output))
  comp_errors.append(absolute_residual_error(test_target, comp_test_output))
  print("Epoch:", i, "SIN Sequential Delta rule error:", comp_errors[-1])

# Plot
ax = plt.gca()

plot_error = True
if plot_error:
  ax.plot(errors)
  ax.plot(comp_errors)
  ax.legend(['Manually placed', 'Competitive learning'])
  plt.ylabel('Absolute residual error')
  plt.xlabel('Epochs')
  plt.xlim([0, 200])
else:
  comp = False
  if comp:
    X = []
    Y = []
    Circles = []
    for node in comp_RBF_Nodes:
      X.append(node.x)
      Y.append(node.y)
      Circles.append(plt.Circle((node.x, node.y), node.variance, color='k', fill=False, alpha= 0.1))

    ax.plot(X, Y, "ro", alpha= 0.1)
    for circle in Circles:
      ax.add_artist(circle)
    ax.scatter(np.asarray(test_input)[:,0], np.asarray(test_input)[:,1])
  else:
    X = []
    Y = []
    Circles = []
    for node in RBF_Nodes:
      X.append(node.x)
      Y.append(node.y)
      Circles.append(plt.Circle((node.x, node.y), node.variance, color='k', fill=False, alpha= 0.1))

    ax.plot(X, Y, "ro", alpha= 0.1)
    for circle in Circles:
      ax.add_artist(circle)
    ax.scatter(np.asarray(test_input)[:,0], np.asarray(test_input)[:,1])

plt.show()


