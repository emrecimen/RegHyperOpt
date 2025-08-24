# ml_optimizer.py
# Core optimization framework

from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
import random
import itertools

# Generate all possible combinations from the search space
def all_combinations(search_space):
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(itertools.product(*values))
    return [list(comb) for comb in combinations]

# Get the indexes of the n largest elements from a list
def largest_indexes(lst, n):
    indexes = []
    for _ in range(n):
        max_element = max(lst)
        max_index = lst.index(max_element)
        indexes.append(max_index)
        lst[max_index] = float('-inf')  # Exclude the largest element
    return indexes

# Main optimization loop
def ml_optimizer(optimize_fn, search_space, iter_num, iter_train_num,
                 train_norm, y_train, valid_norm, y_valid):
    """
    optimize_fn   : callable, function to optimize (must accept kwargs)
    search_space  : dict, parameter search space
    iter_num      : number of iterations
    iter_train_num: number of candidates per iteration
    """
    combinations = all_combinations(search_space)
    parameters, performances = initial_random_search(
        optimize_fn, search_space, iter_train_num, train_norm, y_train, valid_norm, y_valid
    )

    for _ in range(iter_num):
        reg = LinearRegression().fit(parameters, performances)
        predicted_performances = reg.predict(combinations)
        param_indices = largest_indexes(list(predicted_performances), iter_train_num)

        for idx in param_indices:
            params = combinations[idx]
            acc = evaluate_model_from_params(optimize_fn, params, search_space,
                                             train_norm, y_train, valid_norm, y_valid)
            parameters.append(params)
            combinations.pop(idx)
            performances.append(acc)

    return parameters[performances.index(max(performances))]

# Evaluate a function with params
def evaluate_model_from_params(optimize_fn, params, search_space,
                               train_norm, y_train, valid_norm, y_valid):
    param_dict = dict(zip(search_space.keys(), params))
    model = optimize_fn(train_norm, y_train, valid_norm, y_valid, **param_dict)
    _, acc = model.evaluate(valid_norm, y_valid, verbose=0)
    return acc

# Initial random search
def initial_random_search(optimize_fn, search_space, sample_size,
                          train_norm, y_train, valid_norm, y_valid):
    performances = []
    parameters = []
    keys = list(search_space.keys())

    for _ in range(sample_size):
        param_dict = {k: random.choice(v) for k, v in search_space.items()}
        model = optimize_fn(train_norm, y_train, valid_norm, y_valid, **param_dict)
        _, acc = model.evaluate(valid_norm, y_valid, verbose=0)
        performances.append(acc)
        parameters.append([param_dict[k] for k in keys])

    return parameters, performances

# Example optimize function (CNN)
def create_model(train_norm, y_train, valid_norm, y_valid,
                 n_filter, filterSize, poolSize, DenseSize,
                 Beta1, Beta2, LearningRate, BatchSize, Epochs,
                 nofconv, numberofHiderlayer):
    model = Sequential()
    model.add(Conv2D(n_filter, (filterSize, filterSize), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(28, 28, 1)))
    for _ in range(nofconv - 1):
        model.add(Conv2D(n_filter, (filterSize, filterSize), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((poolSize, poolSize)))
    model.add(Flatten())
    for _ in range(numberofHiderlayer):
        model.add(Dense(DenseSize, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(beta_1=Beta1, beta_2=Beta2, learning_rate=LearningRate)
    es = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(train_norm, y_train, batch_size=BatchSize, epochs=Epochs,
              verbose=0, callbacks=[es], validation_data=(valid_norm, y_valid))
    return model
