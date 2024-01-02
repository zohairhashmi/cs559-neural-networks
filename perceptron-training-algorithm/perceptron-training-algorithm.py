import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create function to plot the data points and the line
def create_plot(w0,w1,w2,S1,S2):
    S1 = np.array(S1)
    S2 = np.array(S2)
    x1 = np.linspace(-1,1,100)

    # plot s1 and s2
    plt.scatter(S1[:,0],S1[:,1],marker='o')
    plt.scatter(S2[:,0],S2[:,1],marker='x')
    # plot the new line
    x2 = (-w0 - w1*x1)/w2
    plt.plot(x1,x2)
    plt.legend(['S1','S2','w'])

    plt.show()

# create function to update w using the perceptron learning rule
def update_w(w0,w1,w2,S,S1,S2,alpha):
    for i in range(len(S)):
        a = [1, S[i][0], S[i][1]]
        b = [w0, w1, w2]

        dot = np.dot(a,b) #a.bT

        if dot >= 0:
            if S[i] in np.array(S2):
                w0_updated = w0 - alpha*a[0]
                w1_updated = w1 - alpha*a[1]
                w2_updated = w2 - alpha*a[2]
        else:
            if S[i] in np.array(S1):
                w0_updated = w0 + alpha*a[0]
                w1_updated = w1 + alpha*a[1]
                w2_updated = w2 + alpha*a[2]

    return w0_updated, w1_updated, w2_updated

# create function to report number of misclassified points
def misclassified(w0,w1,w2,S,S1,S2):
    misclassified = 0

    for i in range(len(S)):
        a = [1, S[i][0], S[i][1]]
        b = [w0, w1, w2]

        #a.bT
        dot = np.dot(a,b)

        if dot >= 0:
            if S[i] in np.array(S2):
                misclassified += 1
        else:
            if S[i] in np.array(S1):
                misclassified += 1

    return misclassified

np.random.seed(42)

# pick w0, w1 and w2 uniformally at random
w0 = np.random.uniform(-1/4,1/4)
w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)

print("Line of Separation: w0 + w1x1 + w2x2 = 0")
print("w0 = ", w0)
print("w1 = ", w1)
print("w2 = ", w2)

n = 100
S = np.random.uniform(-1,1,(n,2))

S1 = []
S2 = []

for i in range(len(S)):
    a = [1, S[i][0], S[i][1]]
    b = [w0, w1, w2]

    #a.bT
    dot = np.dot(a,b)

    if dot >= 0:
        S1.append(S[i])
    else:
        S2.append(S[i])

create_plot(w0,w1,w2,S1,S2)


# Random Weights Initializations & Update Weights
alpha = 1 # training parameter

np.random.seed(20)

# pick w0 uniformally at random from [-1,1]
w_0 = np.random.uniform(-1,1)
# pick w1 uniformally at random from [-1,1]
w_1 = np.random.uniform(-1,1)
# pick w2 uniformally at random from [-1,1]
w_2 = np.random.uniform(-1,1)

print("Randomly initialized weights:")
print("w0' = ", w_0)
print("w1' = ", w_1)
print("w2' = ", w_2)

# Experiment 1 : Learning Rate = 1, n = 100
w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr1 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr1.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 1, n = 100 :")
print("w0 = ", w_0_)
print("w1 = ", w_1_)
print("w2 = ", w_2_)


# Experiment 2 : Learning Rate = 10, n = 100
alpha = 10 # training parameter

w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr10 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr10.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 10, n = 100 :")
print("w0 = ", w_0_)
print("w1 = ", w_1_)
print("w2 = ", w_2_)


# Experiment 3 : Learning Rate = 0.1, n = 100
alpha = 0.1 # training parameter

w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr01 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr01.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 0.1, n = 100 :")
print("w0 = ", w_0_)
print("w1 = ", w_1_)
print("w2 = ", w_2_)

lr1_x = range(0, len(data_lr1))
lr1_y = [row[3] for row in data_lr1]

lr10_x = range(0, len(data_lr10))
lr10_y = [row[3] for row in data_lr10]

lr01_x = range(0, len(data_lr01))
lr01_y = [row[3] for row in data_lr01]

# plot 3 subplots for each learning rate
plt.figure(figsize=(20,5))


plt.subplot(1,3,1)
plt.plot(lr10_x, lr10_y)
plt.title('Learning Rate = 10')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.subplot(1,3,2)
plt.plot(lr1_x, lr1_y)
plt.title('Learning Rate = 1')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.subplot(1,3,3)
plt.plot(lr01_x, lr01_y)
plt.title('Learning Rate = 0.1')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.show()


############ Repeating Experiemnts with n = 1000

np.random.seed(42)

# pick w0 uniformally at random from [-1/4,1/4]
w0 = np.random.uniform(-1/4,1/4)

# pick w1 uniformally at random from [-1,1]
w1 = np.random.uniform(-1,1)

# pick w2 uniformally at random from [-1,1]
w2 = np.random.uniform(-1,1)

print("Line of Separation: w0 + w1x1 + w2x2 = 0")
print("w0 = ", w0)
print("w1 = ", w1)
print("w2 = ", w2)

# pick n = 100 vectors x1,...,xn uniformally at random on [-1,1]^2 and assign it to S
n = 1000
S = np.random.uniform(-1,1,(n,2))

S1 = []
S2 = []

for i in range(len(S)):
    a = [1, S[i][0], S[i][1]]
    b = [w0, w1, w2]

    #a.bT
    dot = np.dot(a,b)

    if dot >= 0:
        S1.append(S[i])
    else:
        S2.append(S[i])

np.random.seed(20)

# pick w0 uniformally at random from [-1,1]
w_0 = np.random.uniform(-1,1)
# pick w1 uniformally at random from [-1,1]
w_1 = np.random.uniform(-1,1)
# pick w2 uniformally at random from [-1,1]
w_2 = np.random.uniform(-1,1)

print("Randomly initialized weights:")
print("w0' = ", w_0)
print("w1' = ", w_1)
print("w2' = ", w_2)

# Experiment 1 : Learning Rate = 1, n = 1000
alpha = 1 # training parameter
w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr1_n1000 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr1_n1000.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 1, n = 100 :")
print("w0 = ", w_0)
print("w1 = ", w_1)
print("w2 = ", w_2)


# Experiment 2 : Learning Rate = 10, n = 1000
alpha = 10 # training parameter

w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr10_n1000 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr10_n1000.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 10, n = 100 :")
print("w0 = ", w_0)
print("w1 = ", w_1)
print("w2 = ", w_2)


# Experiment 3 : Learning Rate = 0.1, n = 1000
alpha = 0.1 # training parameter

w_0_, w_1_, w_2_ = w_0, w_1, w_2

misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
misclassified_points_list = []
counter = 0

data_lr01_n1000 = [[w_0, w_1, w_2, misclassified_points]]

while misclassified_points != 0:
    # update w using the perceptron learning rule
    w_0_, w_1_, w_2_ = update_w(w_0_,w_1_,w_2_,S,S1,S2,alpha)
    
    # report the number of misclassified points
    misclassified_points = misclassified(w_0_,w_1_,w_2_,S,S1,S2)
    misclassified_points_list.append(misclassified_points)
    counter += 1

    data_lr01_n1000.append([w_0_, w_1_, w_2_, misclassified_points])

# print final updated weights
print("Final weights for learning rate = 0.1, n = 100 :")
print("w0 = ", w_0)
print("w1 = ", w_1)
print("w2 = ", w_2)

lr1_n1000_x = range(0, len(data_lr1_n1000))
lr1_n1000_y = [row[3] for row in data_lr1_n1000]

lr10_n1000_x = range(0, len(data_lr10_n1000))
lr10_n1000_y = [row[3] for row in data_lr10_n1000]

lr01_n1000_x = range(0, len(data_lr01_n1000))
lr01_n1000_y = [row[3] for row in data_lr01_n1000]

# plot 3 subplots for each learning rate
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
plt.plot(lr10_n1000_x, lr10_n1000_y)
plt.title('Learning Rate = 10, n = 1000')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.subplot(1,3,2)
plt.plot(lr1_n1000_x, lr1_n1000_y)
plt.title('Learning Rate = 1, n = 1000')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.subplot(1,3,3)
plt.plot(lr01_n1000_x, lr01_n1000_y)
plt.title('Learning Rate = 0.1, n = 1000')
plt.xlabel('Epoch')
plt.ylabel('Misclassified Points')

plt.show()

