import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def tanh_activation(x):
    return np.tanh(x)

def linear_activation(x):
    return x

# Define the backpropagation algorithm
def forward_pass(x, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = tanh_activation(hidden_input)
    network_output = linear_activation(np.dot(hidden_output, weights_hidden_output) + bias_output)
    return hidden_input, hidden_output, network_output

def compute_mse(network_output, d):
    return np.mean(np.square(d - network_output))

def backward_pass(x, d, weights_hidden_output, hidden_input, network_output):
    output_error = d - network_output
    hidden_error = output_error.dot(weights_hidden_output.T) * (1 - np.tanh(hidden_input) ** 2)
    return output_error, hidden_error

# Define function to update weights and biases
def update_weights(x, learning_rate, output_error, hidden_error, hidden_output, weights_hidden_output, weights_input_hidden, bias_output, bias_hidden):
    weights_hidden_output += learning_rate * hidden_output.T.dot(output_error)
    weights_input_hidden += learning_rate * x * hidden_error
    bias_output += learning_rate * output_error
    bias_hidden += learning_rate * hidden_error
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Step 1: Generate random data
np.random.seed(42)
n = 300
x = np.random.rand(n)
v = np.random.uniform(-1/10, 1/10, n)
d = np.sin(20 * x) + 3 * x + v

# Step 2: Initialize weights
input_size = 1
hidden_size = 24
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size) # 1x24
weights_hidden_output = np.random.randn(hidden_size, output_size) # 24x1
bias_hidden = np.zeros((1, hidden_size)) # 1x24
bias_output = np.zeros((1, output_size)) # 1x1

# Step 3: Train the network
epochs = 10000
lr = 0.05

mse_history = []

# create copies of the weights and biases
W0 = weights_input_hidden.copy()
W1 = weights_hidden_output.copy()
b0 = bias_hidden.copy()
b1 = bias_output.copy()


for epoch in range(epochs):
    mse = 0

    for i in range(n):
        # forward pass
        layer1_output, layer2_output, network_output = forward_pass(x[i], W0, W1, b0, b1)
        # compute error
        mse += compute_mse(network_output, d[i])
        # backward pass
        output_error, hidden_error = backward_pass(x[i], d[i], W1, layer1_output, network_output)
        # update weights and biases
        W0, W1, b0, b1 = update_weights(x[i], lr, output_error, hidden_error, layer2_output, W1, W0, b1, b0)

    # Calculate mean MSE for the epoch
    mse /= n
    mse_history.append(mse)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {mse}")

    # modify learning rate if MSE is increasing
    if epoch > 0 and mse_history[-1] > mse_history[-2]:
        lr *= 0.9

    # stop if change in MSE is small
    if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < 1e-6:
        print(f"Converged at epoch {epoch}, MSE: {mse}")
        break

# get the output of the trained network using final weights and biases
y = []
for i in range(n):
    _, _, network_output = forward_pass(x[i], W0, W1, b0, b1)
    y.append(network_output[0][0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the number of epochs vs MSE
ax1.plot(range(len(mse_history)), mse_history)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.set_title('Backpropagation Training')

# Plot the generated data and the trained network output
ax2.scatter(x, d, marker='.', label='d')
ax2.scatter(x, y, marker= '+', color='red', label='f(x, w0)')
ax2.set_xlabel('x')
ax2.set_ylabel('d')
ax2.set_title('Original vs Network Output')
ax2.legend()

plt.show()

exit()