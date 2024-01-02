## Homework 1 - Question 2
## Zohair Hashmi - UIN : 668913771

import numpy as np
import matplotlib.pyplot as plt

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def layer1(x, y):
    n1 = x - y + 1
    n2 = -x - y + 1
    n3 = -x
    return step(n1), step(n2), step(n3)
   
def layer2(n1, n2, n3):
    n4 = n1 + n2 - n3 - 1.5
    return step(n4) 

def output(node):
    if node == 1:
        return 'r'
    else:
        return 'b'

def trainNN(x, y):
    n1, n2, n3 = layer1(x, y)
    n4 = layer2(n1, n2, n3)
    return output(n4)

def plot_figure(x, y, z):
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s = 2, c=z) # Plot the points

    plt.plot([0], [0], 'ro', label='Class 1')
    plt.plot([0], [0], 'bo', label='Class 2')
    plt.legend(loc='upper right')
    
    plt.plot([0, 2], [1, -1], 'k-', lw=2) # plot decision boundary from point 0,1 to 2,-1
    plt.plot([0, 0], [1, -2], 'k-', lw=2) # plot decision boundary from point 0,1 to 0, -2
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True, which='both') 
    
    plt.show()


# Generating 1000 points uniformly at random from the square [-2, 2]^2
x = np.random.uniform(-2, 2, 1000)
y = np.random.uniform(-2, 2, 1000)

z = []

# train NN to obtain output class color ('r' or 'b')
for i in range(1000):
    z.append(trainNN(x[i], y[i]))

# plot the points
plot_figure(x, y, z)

# Red  : x >= 0 , y + x -1 < 0
# Blue : x < 0  , y + x -1 >= 0

# print decision boundary equation
print('\n')
print('Equations of the decision boundaries:')
print('x = 0 and y + x - 1 = 0\n')
print('If x >= 0 and y + x - 1 < 0, then the point is classified as red.')
print('If x < 0 or y + x - 1 >= 0, then the point is classified as blue.')
print('\n')

exit()