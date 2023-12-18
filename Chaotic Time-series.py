## Leonard Smedenman
# Chaotic Time-series

import numpy as np
import matplotlib.pyplot as plt


def reservoir_update(r, x, weights_reservoir, weights_inputs):
    r = np.tanh(np.dot(weights_reservoir, r) + np.dot(weights_inputs, x))
    return r


def prediction_update(r, outputs, weights_reservoir, weights_inputs):
    delta_r = np.tanh(np.dot(weights_reservoir, r) + np.dot(weights_inputs, outputs))
    return delta_r


def train_output_weights(x, y, k):
    identity_matrix = np.identity(x.shape[1])
    weights_output = np.linalg.inv(x.T @ x + k * identity_matrix) @ x.T @ y
    return weights_output


def predict_continuation(r_initial, weights_reservoir, weights_inputs, weights_X, weights_Y, weights_Z, steps=500):
    r = r_initial
    predictions_X = []
    predictions_Y = []
    predictions_Z = []

    for _ in range(steps):
        output_X = r @ weights_X
        output_Y = r @ weights_Y
        output_Z = r @ weights_Z

        outputs = np.array([output_X, output_Y, output_Z])
        r = prediction_update(r, outputs, weights_reservoir, weights_inputs)

        predictions_X.append(output_X)
        predictions_Y.append(output_Y)
        predictions_Z.append(output_Z)

    return np.array(predictions_X), np.array(predictions_Y), np.array(predictions_Z)


test_data = np.loadtxt('test-set-5.csv', delimiter=',')
training_data = np.loadtxt('training-set.csv', delimiter=',')

num_inputs = 3
num_reservoir_neurons = 500
ridge_parameter = 0.01

mean = 0
var_input = 0.002
var_reservoir = 2 / 500

weights_inputs = np.random.normal(mean, np.sqrt(var_input), (num_reservoir_neurons, num_inputs))
weights_reservoir = np.random.normal(mean, np.sqrt(var_reservoir), (num_reservoir_neurons, num_reservoir_neurons))

r_train = []
targets_train = []
r = np.zeros(num_reservoir_neurons)

for t in range(training_data.shape[1] - 1):
    x_train = training_data[:, t]
    r = reservoir_update(r, x_train, weights_reservoir, weights_inputs)
    r_train.append(r.copy())
    targets_train.append(training_data[:, t + 1])

r_train = np.array(r_train)
targets_train = np.array(targets_train)

weights_X = train_output_weights(r_train, targets_train[:, 0], ridge_parameter)
weights_Y = train_output_weights(r_train, targets_train[:, 1], ridge_parameter)
weights_Z = train_output_weights(r_train, targets_train[:, 2], ridge_parameter)

r_test = np.zeros(num_reservoir_neurons)
for t in range(test_data.shape[1] - 1):
    x_test = test_data[:, t]
    r_test = reservoir_update(r_test, x_test, weights_reservoir, weights_inputs)

predicted_data_X, predicted_data_Y, predicted_data_Z = predict_continuation(
    r_test, weights_reservoir, weights_inputs, weights_X, weights_Y, weights_Z, steps=500)

print(np.shape(predicted_data_Y))
np.savetxt('prediction.csv', predicted_data_Y)

figure_2D = plt.figure()
plt.plot(predicted_data_Y)

figure_lorenz = plt.figure()
ax = figure_lorenz.add_subplot(111, projection='3d')

ax.plot(predicted_data_X, predicted_data_Y, predicted_data_Z, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("3D Plot of Lorenz Attractor")

plt.show()
