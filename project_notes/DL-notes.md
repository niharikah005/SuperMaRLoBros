# Binary classification:

when the output is either a 0 or a 1, the classification is called as binary classification.
This is the base for logistic regression. There are some important terms to take note of:

m = number of test examples.
nx = number of features in the feature vector
x = feature vector
X = column vector that has all the examples for a single feature vector in m rows.

# Logistic Regression:

Logistic regression is used when we plot the prob of a feature combo to provide either 0 as the output or one as the output.
it uses the following formula: y hat = sigmoid((w)T * x + b), where b is a real number (weight) and x is the feature vector. wT is the parameter. The sigmoid function limits these results to be within 0 to 1, to make sure that we are able to get the right probability.

## Loss function: 
This function checks if the value of y hat is close enough to the provided y value. This helps in knowing the accuracy of the algorithm.
checks error in a single training example

## cost function:
This function avgs all the loss functions over all the training examples. This is the total cost.

# Gradient descent:
This is a method that tries to find parameters w and b such that they are minimising the cost function.