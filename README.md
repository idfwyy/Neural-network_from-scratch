# Neural Network from Scratch — MNIST Digit Recognition

A 2-layer neural network built entirely from scratch in NumPy, with **no machine learning framework** for the model itself. It recognizes handwritten digits (0–9) from the MNIST dataset and reaches **95.64% test accuracy**.

I built this to understand how neural networks actually work under the hood — implementing the forward pass, backpropagation, and gradient descent by hand rather than relying on a framework to hide them.

## What it implements from scratch
- Forward pass (784 → 64 → 10) using matrix multiplication
- ReLU activation
- Softmax (raw scores → probabilities)
- Cross-entropy loss
- Backpropagation (gradients computed by hand)
- Gradient descent training loop

## Result
- **95.64% accuracy** on the MNIST test set (data the network never trained on)

## Tech
- Python, NumPy
- TensorFlow is used *only* to load the MNIST dataset — not for the network itself.

## How to run
```bash
pip install numpy tensorflow
python neural_network.py
```

The script trains the network for 700 epochs, printing the loss as it drops, then reports the final test accuracy.

---
