## Leonard Smedenman
# Perceptron with one hidden layer

import numpy as np
import pandas as pd


def load_and_normalize_data(training_file, validation_file):
    training_data = pd.read_csv(training_file).to_numpy()
    validation_data = pd.read_csv(validation_file).to_numpy()

    x_training = training_data[:, :2]
    target_training = training_data[:, 2]

    x_validation = validation_data[:, :2]
    target_validation = validation_data[:, 2]

    #normalize
    x_training = (x_training - np.mean(x_training, axis=0)) / np.std(x_training, axis=0)
    x_validation = (x_validation - np.mean(x_validation, axis=0)) / np.std(x_validation, axis=0)

    return x_training, target_training, x_validation, target_validation


def calculate_classification_error(outputs, target_validation):
    n_patterns_in_validation_set = len(target_validation)
    return 1 / (2 * n_patterns_in_validation_set) * np.sum(np.abs(np.sign(outputs) - target_validation))


def b(theta, w, input_vector):
    return np.dot(w, input_vector) - theta


def g_prime(b):
    return 1 - np.tanh(b) ** 2


def forward_propagation(x, weights_hidden, weights_output, threshold_hidden, threshold_output):
    V1 = np.tanh(b(threshold_hidden, weights_hidden, x))
    output = np.tanh(b(threshold_output, weights_output, V1))
    return V1, output


def backpropagation(x, target, weights_hidden, weights_output, threshold_hidden, threshold_output, learning_rate):
    V1, output = forward_propagation(x, weights_hidden, weights_output, threshold_hidden, threshold_output)

    output_error = (target - output) * g_prime(b(threshold_output, weights_output, V1))
    dTheta_output = output_error
    dW_output = np.dot(output_error, V1.T)

    hidden_error = np.dot(weights_output.T, output_error) * g_prime(b(threshold_hidden, weights_hidden, x))
    dTheta_hidden = hidden_error
    dW_hidden = np.dot(hidden_error, x.T)

    weights_hidden += learning_rate * dW_hidden
    weights_output += learning_rate * dW_output
    threshold_hidden -= learning_rate * dTheta_hidden
    threshold_output -= learning_rate * dTheta_output

    return weights_hidden, weights_output, threshold_hidden, threshold_output


x_training, target_training, x_validation, target_validation = load_and_normalize_data('training_set.csv', 'validation_set.csv')
n = len(x_training)
m = len(x_validation)

learning_rate = 0.02
M1 = 50

#initialize weights & thresholds
weights_hidden = np.random.normal(0, 1, [M1, 2])
weights_output = np.random.normal(0, 1, [1, M1])
threshold_hidden = np.zeros([M1, 1])
threshold_output = np.zeros([1,1])


epoch = 1
accepted_validation_error = False
while accepted_validation_error == False:
    #stocastic approach
    for _ in range(n):
        mu = np.random.randint(n)
        x = x_training[mu].reshape(-1, 1)
        t = target_training[mu]

        weights_hidden, weights_output, threshold_hidden, threshold_output = backpropagation(x, t, weights_hidden, weights_output,
                                                                                             threshold_hidden, threshold_output, learning_rate)
    outputs = np.zeros(m)

    for k in range(m):
        input_validation = x_validation[k].reshape(-1, 1)
        V1_validation, outputs[k] = forward_propagation(input_validation, weights_hidden, weights_output,
                                                        threshold_hidden, threshold_output)

    validation_error = calculate_classification_error(outputs, target_validation)
    print(f"Epoch={epoch}, Validation Error={validation_error}")

    if validation_error < 0.12:
        accepted_validation_error = True
        break

    epoch += 1


np.savetxt('w1.csv', weights_hidden, delimiter=',')
np.savetxt('w2.csv', weights_output.T, delimiter=',')
np.savetxt('t1.csv', threshold_hidden, delimiter=',')
np.savetxt('t2.csv', threshold_output, delimiter=',')


