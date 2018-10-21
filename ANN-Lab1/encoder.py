import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# "1 of N coding" the input, i.e. ((-1 1 1 1) (1 -1 1 1) (1 1 -1 1) (1 1 1 -1))
data = [[-1] * 8 for _ in range(8)]
for i in range(0, 8):
    data[i][i] = 1
data = np.array(data)

def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

# Two-layer perceptron
input_size = 8
output_size = 8
hidden_layer_size = 3
NUM_POINTS = 8
learning_rate = 0.005

# Initiate weights using small random numbers drawn from the normal distribution with zero mean
mu, sigma = 0, 0.1

v = []
dv = []
for x in range(0, hidden_layer_size):
    v.append(np.random.normal(mu, sigma, input_size + 1)) # + 1
    dv.append(np.random.normal(mu, sigma, input_size + 1)) # + 1

v = np.array(v)
dv = np.array(dv)

w = []
dw = []
for x in range(0, output_size):
    w.append(np.random.normal(mu, sigma, hidden_layer_size + 1)) # + 1
    dw.append(np.random.normal(mu, sigma, hidden_layer_size + 1)) # + 1

w = np.array(w)
dw = np.array(dw)
alpha = 0.9

pat = np.append(data, [[1] * NUM_POINTS], 0)

prev_error = 0
for x in range(0, 1000000):
    # Forward pass
    h_input = np.dot(v, np.append(data, [[1] * NUM_POINTS], 0))
    h_output = np.append(phi( h_input ), [[1] * NUM_POINTS], 0)
    o_input = np.dot(w, h_output)
    o_output = phi( o_input )

    # Backward pass
    delta_o = (o_output - np.asarray(data)) * ((1 + o_output) * (1 - o_output)) * 0.5
    delta_h = (np.dot(np.transpose(w), delta_o)) * ((1 + h_output) * (1 - h_output)) * 0.5
    delta_h = np.delete(delta_h, -1, 0)
    
    # Weight update
    dv = (dv * alpha) - (np.dot(delta_h, np.transpose(pat))) * (1 - alpha)
    dw = (dw * alpha) - (np.dot(delta_o, np.transpose(h_output))) * (1 - alpha)
    v = v + dv * learning_rate
    w = w + dw * learning_rate

    error = np.sum(np.subtract(data, o_output) ** 2) / len(data)
    print("MSE: " + str(error), end='\r')
    #print(error)
    if abs(error - prev_error) < 0.000001 and np.array_equal(data, np.around(o_output)):
        print("Converged at error: " + str(error))
        print("Hidden layer weights:")
        print(np.transpose(np.sign(v)))
        break
    prev_error = error
