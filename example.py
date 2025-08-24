# example.py
# Simple demo for ml_optimizer

from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from regHyperOpt import ml_optimizer, create_model

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# Normalize
train_norm = x_train.astype('float32') / 255.0
test_norm = x_test.astype('float32') / 255.0
valid_norm = x_valid.astype('float32') / 255.0

# One-hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_valid = np_utils.to_categorical(y_valid)

print(f"Train: {train_norm.shape}, Test: {test_norm.shape}, Valid: {valid_norm.shape}")

# Search space
search_space = {
    'n_filter': [8, 16],
    'filterSize': [3],
    'poolSize': [2],
    'DenseSize': [32],
    'Beta1': [0.99],
    'Beta2': [0.99],
    'LearningRate': [0.001],
    'BatchSize': [16],
    'Epochs': [3],  # small for demo
    'nofconv': [1, 2],
    'numberofHiderlayer': [1]
}

# Run optimizer
best_params = ml_optimizer(
    optimize_fn=create_model,
    search_space=search_space,
    iter_num=1,
    iter_train_num=1,
    train_norm=train_norm, y_train=y_train,
    valid_norm=valid_norm, y_valid=y_valid
)

print("Best Parameters:", best_params)
