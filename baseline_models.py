from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


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

print("Training", training_data.shape, training_labels.shape)
print("Testing", testing_data.shape, testing_labels.shape)

training_samples = training_data[:1000]
sample_labels = training_labels[:1000]

# KNN
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(training_samples, sample_labels)

print("KNN Training Accuracy", neigh.score(training_samples, sample_labels))
print("KNN Testing Accuracy", neigh.score(testing_data, testing_labels))

# Naive Bayes
nb = MultinomialNB()
nb.fit(training_samples, sample_labels)

print("Naive Bayes Training Accuracy", nb.score(training_samples, sample_labels))
print("Naive Bayes Accuracy", nb.score(testing_data, testing_labels))
