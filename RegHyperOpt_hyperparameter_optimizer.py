# Created by Emre Cimen

from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.optimizers import Adam
import itertools

# Generate all possible combinations from the search space
def all_combinations(search_space):
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(itertools.product(*values))
    combinations_list = [list(comb) for comb in combinations]
    return combinations_list

# Get the indexes of the n largest elements from a list
def largest_indexes(lst, n):
    indexes = []
    for _ in range(n):
        max_element = max(lst)
        max_index = lst.index(max_element)
        indexes.append(max_index)
        lst[max_index] = float('-inf')  # Remove the largest element
    return indexes

# Main optimization loop using regression to predict performance
def ml_optimizer(iter_num, iter_train_num):
    combinations = all_combinations(search_space)
    parameters, performances = initial_random_search(search_space, iter_train_num)

    for _ in range(iter_num):
        reg = LinearRegression().fit(parameters, performances)
        predicted_performances = reg.predict(combinations)
        param_indices = largest_indexes(list(predicted_performances), iter_train_num)

        for idx in param_indices:
            acc = evaluate_model_from_params(combinations[idx])
            parameters.append(combinations[idx])
            combinations.pop(idx)
            performances.append(acc)

    return parameters[np.argmax(performances)]

# Create and evaluate a model given a parameter set
def evaluate_model_from_params(params):
    n_filter, filter_size, pool_size, dense_size, beta1, beta2, learning_rate, batch_size, epochs, num_conv, num_hidden = params
    model = create_model(n_filter, filter_size, pool_size, dense_size, beta1, beta2, learning_rate, batch_size, epochs, num_conv, num_hidden)
    _, acc = model.evaluate(valid_norm, y_valid, verbose=0)
    return acc

# Perform an initial random parameter search
def initial_random_search(search_space, sample_size):
    performances = []
    parameters = []

    for _ in range(sample_size):
        n_filter = random.choice(search_space['n_filter'])
        filter_size = random.choice(search_space['filterSize'])
        pool_size = random.choice(search_space['poolSize'])
        dense_size = random.choice(search_space['DenseSize'])
        beta1 = random.choice(search_space['Beta1'])
        beta2 = random.choice(search_space['Beta2'])
        learning_rate = random.choice(search_space['LearningRate'])
        batch_size = random.choice(search_space['BatchSize'])
        epochs = random.choice(search_space['Epochs'])
        num_conv = random.choice(search_space['nofconv'])
        num_hidden = random.choice(search_space['numberofHiderlayer'])

        model = create_model(n_filter, filter_size, pool_size, dense_size, beta1, beta2, learning_rate, batch_size, epochs, num_conv, num_hidden)
        _, acc = model.evaluate(valid_norm, y_valid, verbose=0)
        performances.append(acc)
        parameters.append([n_filter, filter_size, pool_size, dense_size, beta1, beta2, learning_rate, batch_size, epochs, num_conv, num_hidden])

    return parameters, performances

# Build and train the CNN model
def create_model(n_filter, filter_size, pool_size, dense_size, beta1, beta2, learning_rate, batch_size, epochs, num_conv, num_hidden):
    model = Sequential()

    model.add(Conv2D(n_filter, (filter_size, filter_size), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(28, 28, 1)))
    for _ in range(num_conv - 1):
        model.add(Conv2D(n_filter, (filter_size, filter_size), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(MaxPooling2D((pool_size, pool_size)))
    model.add(Flatten())

    for _ in range(num_hidden):
        model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(10, activation='softmax'))

    adam = Adam(beta_1=beta1, beta_2=beta2, learning_rate=learning_rate)
    es = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(train_norm, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[es], validation_data=(valid_norm, y_valid))

    return model

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

print(f'Train: X={x_train.shape}, y={y_train.shape}')
print(f'Test: X={x_test.shape}, y={y_test.shape}')
print(f'Valid: X={x_valid.shape}, y={y_valid.shape}')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_valid = np_utils.to_categorical(y_valid)

train_norm = x_train.astype('float32') / 255.0
test_norm = x_test.astype('float32') / 255.0
valid_norm = x_valid.astype('float32') / 255.0

# Define the hyperparameter search space
search_space = {
    'n_filter': [4, 8, 16],
    'filterSize': [1, 3],
    'poolSize': [1, 3],
    'DenseSize': [8, 16, 32],
    'Beta1': [0.90, 0.93, 0.95, 0.97, 0.99],
    'Beta2': [0.90, 0.93, 0.95, 0.97, 0.99],
    'LearningRate': [0.001, 0.01],
    'BatchSize': [8, 16, 32],
    'Epochs': [20, 30],
    'nofconv': [1, 2],
    'numberofHiderlayer': [1, 2]
}

# Example: Direct model training without optimization loop
model = create_model(n_filter=16, filter_size=3, pool_size=3, dense_size=32, beta1=0.99, beta2=0.99, learning_rate=0.001,
                     batch_size=16, epochs=20, num_conv=2, num_hidden=1)
_, acc = model.evaluate(test_norm, y_test, verbose=0)
print(acc)
