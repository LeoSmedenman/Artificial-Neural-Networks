## Leonard Smedenman
# Boolean functions

import itertools
import numpy as np


def activation_function(w, x, theta):
    #calculates the dot product of weights and inputs, subtracts the threshold and returns the sign
    b = np.dot(x, w) - theta
    g = np.sign(b)
    return g


def delta_wj_learning_rules(eta, t, output, x):
    #calculates the weight update based on the learning rate, target, output and input
    delta_wj = eta * (t - output) * x
    return delta_wj


def delta_theta_learning_rules(eta, t, output):
    #calculates the threshold update based on the learning rate, target and output
    delta_theta = -eta * (t - output)
    return delta_theta


def train_perceptron(epochs, weights, inputs, threshold, target, eta):
    #trains the perceptron for # epochs. Updates weights and threshold based on the delta rules.
    for _ in range(epochs):
        output = activation_function(weights, inputs, threshold)
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                delta_wj = delta_wj_learning_rules(eta, target[i], output[i], inputs[i][j])
                weights[j] += delta_wj
            delta_theta = delta_theta_learning_rules(eta, target[i], output[i])
            threshold += delta_theta
    return output


def evaluate_linear_separability(dimensions, eta, n_iterations, epochs, threshold):
    #evaluates linear separability for different dimensions. Generates random target functions and checks if the perceptron can separate them
    for n in dimensions:
        separable_count = 0
        inputs = list(itertools.product([0, 1], repeat=n))
        target_list = []

        ## just for printing purposes
        unique_target_counter = 0
        percentage_linearly_functions = [14/16, 104/256, 1882/65536, 94572/4294967296]

        for i in range(n_iterations):
            target = tuple(np.random.choice([1, -1], 2**n))

            if target not in target_list:
                target_list.append(target)
                unique_target_counter += 1
                weights = np.random.normal(0, np.sqrt(1 / n), n)
                outputs = train_perceptron(epochs, weights, inputs, threshold, target, eta)
                if np.array_equal(outputs, target):
                    separable_count += 1
        print(f"Linearly separable functions for {n} dimensions: {separable_count}. ({100 * separable_count / unique_target_counter}%)")
        print(f"Actual percentage of linearly separable functions for {n} dimensions {100*percentage_linearly_functions[n-2]}%\n")


dimensions = [2, 3, 4, 5]
eta = 0.05
n_iterations = 10**4
epochs = 20
threshold = 0

evaluate_linear_separability(dimensions, eta, n_iterations, epochs, threshold)

