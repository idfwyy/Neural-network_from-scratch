import tensorflow
import numpy as np
from tensorflow.keras.datasets import mnist


"""
I had no prior knowledge of how a neural network works. I learned it step by step,
term by term, and built this network by hand. Claude helped me with the concepts,
forward pass, backprop, loss, and gradients, but I only started writing code once
I actually understood why I was writing it and what it was for. I began with messy,
scattered code and cleaned it up as my understanding grew. A few smaller projects
with Python, pandas, and NumPy beforehand made this a lot easier to build.
"""

# Load the MNIST dataset, split into train and test sets.
# Pixel values are 0-255; I divide by 255 so they sit between 0 and 1, since
# smaller numbers are faster and less error-prone for the network to work with.
# Then I reshape each 28x28 image into a flat row of 784 values (1D), so it's
# easier to feed through the network.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# These are the network's weights and biases, started as small random numbers.
# w1, b1 connect the input to the hidden layer; w2, b2 connect the hidden layer
# to the output. I multiply by 0.01 to keep the starting values small.
w1 = np.random.randn(784, 64) * 0.01
b1 = np.random.randn(64) * 0.01
w2 = np.random.randn(64, 10) * 0.01
b2 = np.random.randn(10) * 0.01

# forward: runs an image through the network. It multiplies the input by w1, adds
# b1, and passes it through ReLU (which turns negatives to 0, keeps positives).
# Then the hidden output goes through w2/b2 and softmax to get the final scores.
# softmax: turns raw output scores into probabilities that add up to 1.
# cross_entropy: measures how wrong the network is using -log(correct probability),
# averaged over all 60k images. one_hot: turns each label into the "correct" answer
# array (a 1 at the right digit, 0 everywhere else).
def forward(x):
    layer1 = x @ w1 + b1
    relu = np.maximum(0, layer1)
    layer2 = relu @ w2 + b2
    probab = softmax(layer2)
    return probab, relu, layer1, layer2

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy(probs, labels):
    corr_prob = probs[range(len(labels)), labels]
    loss = -np.log(corr_prob)
    return np.mean(loss)

def one_hot(labels):
    y_true = np.zeros((len(labels), 10))
    y_true[range(len(labels)), labels] = 1
    return y_true

# The learning rate controls how big a step the network takes each round.
# I loop 700 times, passing all 60k images through repeatedly to train it.
learning_rate = 0.5
N = len(X_train)
y_true = one_hot(y_train)

# Backprop is the core of learning. After the forward pass gives a prediction, I
# find how wrong it was (output_error = probs - y_true). Then I trace that error
# backwards through the network to figure out how much each weight and bias caused
# it. That "blame" for each weight is called its gradient (dw1, db1, dw2, db2).
# The error moves back through w2 first, then through ReLU (only neurons that were
# active pass error back), then to w1. Once I have the gradients, gradient descent
# nudges every weight and bias a small step in the direction that lowers the loss.
# Repeating this loop is what makes the network slowly get better.
for epoch in range(700):
    # FORWARD
    probs, relu, layer1, layer2 = forward(X_train)

    # LOSS (print every 10 epochs to watch it drop)
    loss = cross_entropy(probs, y_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # BACKPROP (trace the error back, get the gradients)
    output_error = probs - y_true
    dw2 = relu.T @ output_error / N
    db2 = np.sum(output_error, axis=0) / N
    hidden_error = (output_error @ w2.T) * (layer1 > 0)
    dw1 = X_train.T @ hidden_error / N
    db1 = np.sum(hidden_error, axis=0) / N

    # UPDATE (gradient descent: nudge weights to reduce loss)
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

# Finally, I check accuracy on the test set, data the network has never seen,
# to see how many digits out of 100 it predicts correctly.
test_probs, _, _, _ = forward(X_test)
predictions = np.argmax(test_probs, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")