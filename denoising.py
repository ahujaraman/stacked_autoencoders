import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_der(x):
    s = 1. / (1 + np.exp(-x))
    return s * (1. - s)


def one_hot_encode_labels(labels):
    n_unique = len(set(labels))
    output = []
    for label in labels:
        a = np.zeros(n_unique)
        a[label] = 1
        output.append(a)

    return np.asarray(output)


def add_noise(data_original, noise_percentage=20):
    data = np.copy(data_original)
    n_columns = data.shape[1]
    n_noisy = int(noise_percentage * n_columns / 100)
    for i in range(data.shape[0]):
        noisy = np.random.choice(np.arange(n_columns), n_noisy, replace=False)
        data[i][noisy] = 0
    return data


def encode(x, params):
    return sigmoid(np.dot(x, params["W1"].T) + params["b1"])


def decode(h, params):
    return sigmoid(np.dot(h, params["W2"].T) + params["b2"])


def cost_function(x, z):
    eps = 1e-10
    return - np.sum((x * np.log(z + eps) + (1. - x) * np.log(1. - z + eps)))


def get_n_samples_per_class(data, labels, n):
    samples = []
    for i in range(10):
        s = data[labels == i]
        s = s[:n]
        samples.append(s)
    return np.asarray(samples).reshape(n * 10, 784)


def calc_gradient(x_batch, y_expected, params):
    
    x_data = x_batch
    p = encode(x_data, params)
    y = decode(p, params)
    cost = np.sum(cost_function(y_expected, y))

    delta1 = y - y_expected

    dW2 = np.sum(np.dot(delta1.T, p), axis=0)
    db2 = np.sum(delta1, axis=0)
    delta2 = np.sum(np.dot(params["W2"].T, delta1.T) * sigmoid_der(p).T, axis=0)
    dW1 = np.sum(np.dot(delta2, x_data), axis=0)
    db1 = np.sum(delta2, axis=0)

    cost /= len(x_batch)
    dW1 /= len(x_batch)
    dW2 /= len(x_batch)
    db1 /= len(x_batch)
    db2 /= len(x_batch)

    return cost, dW1, dW2, db1, db2


def train(X, Y, params, epochs=10, batch_size=128, alpha=0.1):
    costs = []

    batch_num = len(X) // batch_size

    for epoch in range(epochs):
        total_cost = 0.0
        print("Epoch", epoch)

        for i in range(batch_num):
            batch = X[i * batch_size: (i + 1) * batch_size]
            batch_output = Y[i * batch_size: (i + 1) * batch_size]

            cost, gradW1, gradW2, gradb1, gradb2 = calc_gradient(batch, batch_output, params)

            total_cost += cost
            params["W1"] -= alpha * gradW1
            params["W2"] -= alpha * gradW2
            params["b1"] -= alpha * gradb1
            params["b2"] -= alpha * gradb2

        costs.append((1. / batch_num) * total_cost)
        print("Epoch", epoch, "Cost", (1. / batch_num) * total_cost)
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.title("Training - Denoising autoencoder")
    plt.show()


# Training data
print("Reading training data...")
raw_training_data = pd.read_csv("../dataset/fashion-mnist_train.csv")
training_data = np.asarray(raw_training_data.loc[:, raw_training_data.columns != 'label'], dtype=np.float)
training_labels = one_hot_encode_labels(np.asarray(raw_training_data['label']))

print("Reading testing data...")
# Testing data
raw_testing_data = pd.read_csv("../dataset/fashion-mnist_test.csv")
testing_data = np.asarray(raw_testing_data.loc[:, raw_testing_data.columns != 'label'], dtype=np.float)
testing_labels = one_hot_encode_labels(np.asarray(raw_testing_data['label']))

print("Training", training_data.shape, training_labels.shape)
print("Testing", testing_data.shape, testing_labels.shape)

training_data = training_data / 255.0

n_inputs = training_data.shape[1]
n_output = n_inputs
n_hidden = 1000
noise = 10
noise_testing = 10

noisy_training_data = add_noise(training_data, noise_percentage=noise)

params = {"W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
          "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
          "b1": np.zeros((n_hidden)),
          "b2": np.zeros((n_output))}
train(noisy_training_data, training_data, params)

samples_input = get_n_samples_per_class(testing_data, raw_testing_data['label'], 1)
noisy_sample_input = add_noise(samples_input, noise_percentage=noise_testing)
output = decode((encode(noisy_sample_input, params)), params)


for i in range(10):
    f = plt.subplot(3, 10, i + 1)
    plt.imshow(samples_input[i].reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.axis('off')
    f.set_xticklabels([])
    f.set_yticklabels([])

for i in range(10):
    f = plt.subplot(3, 10, 10 + i + 1)
    plt.imshow(noisy_sample_input[i].reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.axis('off')
    f.set_xticklabels([])
    f.set_yticklabels([])

for i in range(10):
    f = plt.subplot(3, 10, 20 + i + 1)
    plt.imshow(output[i].reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.axis('off')
    f.set_xticklabels([])
    f.set_yticklabels([])

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.0, hspace=0.0)
# plt.tight_layout()
plt.show()
