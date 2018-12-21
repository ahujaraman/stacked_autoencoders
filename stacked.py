import numpy as np
import pandas as pd


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_der(x):
    s = 1. / (1 + np.exp(-x))
    return s * (1. - s)


def one_hot_encode_labels(labels):
    n_unique = 10
    output = []
    for label in labels:
        a = np.zeros(n_unique)
        a[label] = 1
        output.append(a)

    return np.asarray(output)


def encode(x, params):
    return sigmoid(np.dot(x, params["W1"].T) + params["b1"])


def decode(h, params):
    return sigmoid(np.dot(h, params["W2"].T) + params["b2"])


def cost_function(x, z):
    eps = 1e-10
    return - np.sum((x * np.log(z + eps) + (1. - x) * np.log(1. - z + eps)))


def calc_gradient(x_batch, y_expected, params):
    data_batch = x_batch
    p = encode(data_batch, params)
    y = decode(p, params)
    cost = np.sum(cost_function(y_expected, y))

    delta1 = y - y_expected

    dw2 = np.sum(np.dot(delta1.T, p), axis=0)
    db2 = np.sum(delta1, axis=0)
    delta2 = np.sum(np.dot(params["W2"].T, delta1.T) * sigmoid_der(p).T, axis=0)
    dw1 = np.sum(np.dot(delta2, data_batch), axis=0)
    db1 = np.sum(delta2, axis=0)

    cost /= len(x_batch)
    dw1 /= len(x_batch)
    dw2 /= len(x_batch)
    db1 /= len(x_batch)
    db2 /= len(x_batch)

    return cost, dw1, dw2, db1, db2


def train(X, Y, params, epochs=1, batch_size=128, alpha=0.1):
    costs = []

    batch_num = len(X) // batch_size

    for epoch in range(epochs):
        total_cost = 0.0

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


def stacked_autoencoder_transform(x, params1, params2, params3):
    L1 = encode(x, params1)
    L2 = encode(L1, params2)
    L3 = encode(L2, params3)

    return L3


def get_n_samples_per_class(data, labels, n):
    samples = []
    label = []
    for i in range(10):
        s = data[labels == i]
        l = labels[labels == i]

        l = l[:n]
        s = s[:n]
        samples.append(s)
        label.append(l)
    return np.asarray(samples).reshape(n * 10, 784), np.asarray(label).reshape(n * 10, 1)


def predict(data, params):
    l1 = encode(data, params)
    output = decode(l1, params)
    return output


def test_accuracy(testing_data, testing_labels, params):
    predicted_output = predict(testing_data, params)

    count = 0
    for i in range(len(testing_data)):
        if np.argmax(predicted_output[i]) == np.argmax(testing_labels[i]):
            count += 1

    acc = count * 100 / len(testing_data)
    return acc


# Training data
print("Reading training data...")
raw_training_data = pd.read_csv("../dataset/fashion-mnist_train.csv")
training_data = np.asarray(raw_training_data.loc[:, raw_training_data.columns != 'label'], dtype=np.float)
training_labels = np.asarray(raw_training_data['label'])

print("Reading testing data...")
# Testing data
raw_testing_data = pd.read_csv("../dataset/fashion-mnist_test.csv")
testing_data = np.asarray(raw_testing_data.loc[:, raw_testing_data.columns != 'label'], dtype=np.float)
testing_labels = np.asarray(raw_testing_data['label'])
testing_labels = one_hot_encode_labels(testing_labels)

print("Training", training_data.shape, training_labels.shape)
print("Testing", testing_data.shape, testing_labels.shape)

training_data = training_data / 255.0
testing_data = testing_data / 255.0

print("Training Layer 1")
# 728 to 500
n_inputs = training_data.shape[1]
n_output = n_inputs
n_hidden = 500
params_layer_1 = {
    "W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
    "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
    "b1": np.zeros((n_hidden)),
    "b2": np.zeros((n_output))}
train(training_data, training_data, params_layer_1, epochs=5)
layer_1_output = encode(training_data, params_layer_1)

print("Training Layer 2")
# 500 to 300
n_inputs = 500
n_output = n_inputs
n_hidden = 300
params_layer_2 = {
    "W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
    "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
    "b1": np.zeros((n_hidden)),
    "b2": np.zeros((n_output))}
train(layer_1_output, layer_1_output, params_layer_2, epochs=5)
layer_2_output = encode(layer_1_output, params_layer_2)

print("Training Layer 3")
# 300 to 100
n_inputs = 300
n_output = n_inputs
n_hidden = 100
params_layer_3 = {
    "W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
    "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
    "b1": np.zeros((n_hidden)),
    "b2": np.zeros((n_output))}
train(layer_2_output, layer_2_output, params_layer_3, epochs=5)


one_sample_per_class_data, one_sample_per_class_labels = get_n_samples_per_class(training_data, training_labels, 1)
five_sample_per_class_data, five_sample_per_class_labels = get_n_samples_per_class(training_data, training_labels, 5)

# One - Hot Encoding
one_sample_per_class_labels = one_hot_encode_labels(one_sample_per_class_labels.tolist())
five_sample_per_class_labels = one_hot_encode_labels(five_sample_per_class_labels.tolist())

# Reducing size using stacked AE
one_sample_reduced_training_data = stacked_autoencoder_transform(one_sample_per_class_data, params_layer_1,
                                                                 params_layer_2, params_layer_3)
five_sample_reduced_training_data = stacked_autoencoder_transform(five_sample_per_class_data, params_layer_1,
                                                                  params_layer_2, params_layer_3)

testing_data = stacked_autoencoder_transform(testing_data, params_layer_1, params_layer_2, params_layer_3)

# Classifier
n_inputs = 100
n_output = 10
n_hidden = 100
params_classifier_one_sample = {
    "W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
    "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
    "b1": np.zeros((n_hidden)),
    "b2": np.zeros((n_output))}

print("Training Classifier with one sample per class")
train(one_sample_reduced_training_data, one_sample_per_class_labels, params_classifier_one_sample, batch_size=1,
      epochs=10)

params_classifier_five_sample = {
    "W1": np.asarray(np.random.uniform(low=(-1. / n_inputs), high=(1. / n_inputs), size=(n_hidden, n_inputs))),
    "W2": np.asarray(np.random.uniform(low=(-1. / n_hidden), high=(1. / n_hidden), size=(n_output, n_hidden))),
    "b1": np.zeros((n_hidden)),
    "b2": np.zeros((n_output))}

print("Training Classifier with five samples per class")
train(one_sample_reduced_training_data, one_sample_per_class_labels, params_classifier_five_sample, batch_size=1,
      epochs=10)

# Testing

# Testing model with one sample per class
acc = test_accuracy(testing_data, testing_labels, params_classifier_one_sample)
print("Test accuracy with one training sample per class", acc)

# Testing model with five samples per class
acc = test_accuracy(testing_data, testing_labels, params_classifier_five_sample)
print("Test accuracy with five training samples per class", acc)
