import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check
from scipy.optimize import fmin_l_bfgs_b as lbfgs

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def cost(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = sigmoid(np.dot(inputs, weights))
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Build a function that returns gradients of training loss using autograd.
cost_grad = grad(cost)

# Check the gradients numerically, just to be safe.
weights = np.array([0.0, 0.0, 0.0])
quick_grad_check(cost, weights)

# Optimize weights using gradient descent.
print "Initial loss:", cost(weights)
momentum = 0
for i in xrange(1000):
    # print cost_grad(weights)
    momentum = cost_grad(weights) + momentum*0.8
    weights -= momentum
    # print cost(weights)

print  "Trained loss:", cost(weights)
print weights

weights = np.array([0.0, 0.0, 0.0])
[x, f, d] = lbfgs(func=cost, x0=weights, fprime=cost_grad)
print x
print f
print d

print cost(np.array([ 4.82414793,-0.91942305, 6.91707966]))
