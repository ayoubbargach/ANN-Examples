from operator import add
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import argparse

def energy(weights, input_pattern):
  #size = range(0, len(input_pattern))
  #return -reduce(add, [weights[i][j] * input_pattern[i] * input_pattern[j] for i in size for j in size])
  return -weights.dot(input_pattern).dot(input_pattern.T)

def plot_energy(energy_list):
  plt.plot(energy_list)
  plt.ylabel('Energy')
  plt.xlabel('Iterations')
  plt.show()

def plot_energies(energy_lists, legend):
  for energy_list in energy_lists:
    plt.plot(energy_list)
  plt.legend(legend);
  plt.ylabel('Energy')
  plt.xlabel('Iterations')
  plt.show()

def energy_at(weights, patterns, what = "patterns"):
  print("ENERGY AT " + what + ":")
  print("====================")
  for pattern in patterns:
    print(str(pattern) + ": " + str(energy(weights, pattern)))

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

def find_attractors(weights):
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
        
  return remove_duplicates(attractors)

def sequential_update(weights, pattern):
  pattern = np.array(pattern)
  out_patterns = [pattern.copy()]
  size = range(0, len(pattern))
  converged = False
  for iters in range(0, 20):
    old_pattern = pattern.copy()
    for j in size:
      sum = 0
      for i in size:
        sum += weights[j][i] * pattern[i]
      pattern[j] = np.sign(sum)
      out_patterns.append(pattern.copy())
    if (pattern == old_pattern).all():
      converged = True
      break
  return pattern, out_patterns, converged


def main():
  parser = argparse.ArgumentParser(description='Do some Energy.')
  parser.add_argument('-p', '--part', type=int, required=True,
                     help='which part of the assignment to run')
  args = parser.parse_args()

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

  # What is the energy at the di erent attractors?
  if args.part == 1:
    attractors = find_attractors(weights)
    print("ATTRACTORS: ", len(attractors))
    print(attractors)
    energy_at(weights, attractors, "attractors")

  # What is the energy at the points of the distorted patterns?
  elif args.part == 2:
    energy_at(weights, [x1d, x2d, x3d], "distorted input patterns")

  # Follow how the energy changes from iteration to iteration when you use the sequential update rule to approach an attractor.
  elif args.part == 3:
    (_, x1d_seq, _) = sequential_update(weights, x1d)
    (_, x2d_seq, _) = sequential_update(weights, x2d)
    (_, x3d_seq, _) = sequential_update(weights, x3d)
    plot_energies(list(map(lambda x: list(map(lambda y: energy(weights, y), x)), [x1d_seq, x2d_seq, x3d_seq])),
      ['x1d', 'x2d', 'x3d'])

  # Generate a weight matrix by setting the weights to normally distributed random numbers, and try iterating an arbitrary starting state. What happens?
  elif args.part == 4:
    converged_count = 0
    for i in range(0, 100):
      arbitrary_weights = np.random.normal(0, 1, (8, 8))
      arbitrary_state = np.ones(8)
      (_, arbitrary_seq, converged) = sequential_update(arbitrary_weights, arbitrary_state)
      if converged:
        converged_count += 1
    plot_energy(list(map(lambda x: energy(arbitrary_weights, x), arbitrary_seq)))
    print("converged: " + str(converged_count) + " out of 100 times")

  # Make the weight matrix symmetric (e.g. by setting w=0.5*(w+w')). What happens now? Why?
  elif args.part == 5:
    converged_count = 0
    for i in range(0, 100):
      arbitrary_weights = np.random.normal(0, 1, (8, 8))
      arbitrary_weights = np.dot(0.5, arbitrary_weights + arbitrary_weights.T)
      arbitrary_state = np.ones(8)
      (_, arbitrarier_seq, converged) = sequential_update(arbitrary_weights, arbitrary_state)
      if converged:
        converged_count += 1
    plot_energy(list(map(lambda x: energy(arbitrary_weights, x), arbitrarier_seq)))
    print("converged: " + str(converged_count) + " out of 100 times")


if __name__ == "__main__":
  main()