import numpy as np


def perceptron_activation(n, w, x, theta):
    b = sum(w[j] * x[j] for j in range(n)) - theta
    g = np.sign(b)
    return g


def delta_wj_learning_rules(eta, t, big_O, x):
    delta_wj = eta * (t - big_O) * x
    return delta_wj


def delta_theta_learning_rules(eta, t, big_O):
    delta_theta = -eta * (t - big_O)
    return delta_theta


def generate_boolean_function(n):
    return tuple(np.random.randint(0, 2, size=n))  # Convert to tuple


def train_perceptron(inputs, outputs, eta, epochs=20):
    n = len(inputs[0])
    w = np.random.normal(0, np.sqrt(1 / n), n)
    theta = 0

    for _ in range(epochs):
        for i in range(len(inputs)):
            x = inputs[i]
            t = outputs[i]

            big_O = perceptron_activation(n, w, x, theta)

            for j in range(n):
                delta_wj = delta_wj_learning_rules(eta, t, big_O, x[j])
                w[j] += delta_wj

            delta_theta = delta_theta_learning_rules(eta, t, big_O)
            theta += delta_theta

    return w, theta


def is_linearly_separable(inputs, outputs):
    w, theta = train_perceptron(inputs, outputs, eta=0.05)

    for i in range(len(inputs)):
        predicted_output = perceptron_activation(len(inputs[0]), w, inputs[i], theta)
        if predicted_output != outputs[i]:
            return False

    return True


dimensions = [2, 3, 4, 5]
iterations = 10 ** 4
linearly_separable_counts = {}

for n in dimensions:
    linearly_separable_count = 0
    unique_functions = set()  # To keep track of unique Boolean functions

    while linearly_separable_count < iterations:
        inputs = [generate_boolean_function(n) for _ in range(4)]
        outputs = np.array([1, -1, -1, -1])  # Choose linearly separable outputs

        if inputs not in unique_functions:  # Check if the same function was used before
            unique_functions.add(inputs)

            if is_linearly_separable(inputs, outputs):
                linearly_separable_count += 1

    linearly_separable_counts[n] = linearly_separable_count

print("Linearly Separable Counts:", linearly_separable_counts)
