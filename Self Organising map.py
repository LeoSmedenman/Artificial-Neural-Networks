## Leonard Smedenman
# Self-organizing map

import numpy as np
import matplotlib.pyplot as plt


def neighborhood_function(distance, sigma):
    return np.exp(-1 / (2 * sigma ** 2) * distance ** 2)


def update_weights(pattern, weight_matrix, learning_rate, sigma, winning_neuron):
    x, y = np.meshgrid(np.arange(weight_matrix.shape[0]), np.arange(weight_matrix.shape[1]))
    d = np.sqrt((x - winning_neuron[0]) ** 2 + (y - winning_neuron[1]) ** 2)
    h = neighborhood_function(d, sigma)

    weight_matrix += learning_rate * np.expand_dims(h, axis=2) * (pattern - weight_matrix)
    return weight_matrix


def visualize_neurons(neurons, targets, title):
    for i, (x, y) in enumerate(neurons):
        target_class = int(targets[i])
        color = ['r', 'g', 'b'][target_class]
        plt.scatter(x, y, color=color, label=f'Class {target_class}')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.xlabel('Grid X-coordinate')
    plt.ylabel('Grid Y-coordinate')
    plt.grid(True)


inputs = np.genfromtxt('iris-data.csv', delimiter=',')
targets = np.genfromtxt('iris-labels.csv', delimiter=',')

output_size = (40, 40, 4)
initial_learning_rate = 0.1
initial_sigma = 10.0
decay_learning_rate = 0.01
decay_sigma = 0.05
num_epochs = 10

weight_matrix = np.random.rand(*output_size)

max_values = inputs.max(axis=0)
standardized_inputs = inputs / max_values

sigma = initial_sigma
learning_rate = initial_learning_rate

initial_winning_neurons = []
final_winning_neurons = []

for epoch in range(1, num_epochs + 1):
    for pattern in standardized_inputs:
        distances = np.linalg.norm(weight_matrix - pattern, axis=2)
        min_distance_index = np.argmin(distances)
        winning_neuron = np.unravel_index(min_distance_index, (output_size[0], output_size[1]))

        if epoch == 1:
            initial_winning_neurons.append(winning_neuron)
        elif epoch == num_epochs:
            final_winning_neurons.append(winning_neuron)

        weight_matrix = update_weights(pattern, weight_matrix, learning_rate, sigma, winning_neuron)

    learning_rate *= np.exp(-decay_learning_rate * epoch)
    sigma *= np.exp(-decay_sigma * epoch)


plt.subplot(1, 2, 1)
visualize_neurons(initial_winning_neurons, targets, 'Winning Neurons Before Training')

plt.subplot(1, 2, 2)
visualize_neurons(final_winning_neurons, targets, 'Winning Neurons After Training')

plt.show()

