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
the formula is as follows: w = w - alpha (grad cost func wrt w) where alpha is the learning rate of the network
if minimising both w and b at the same time use the same formula but partially derivate cost func wrt parameters.

(for both cases, the slope is the division of small values of height/width or y/x if a graph has been created)

# Computational graph:
This is a type of graph that links variables to the final output they produce after a certain operation has been performed on them.
This a great way of breaking down complex calculations and finding easier derivatives to take when needed. One step of backpropagation on this graph helps in finding partial derivatives

The steps to find regression then are simple: first find the values of z ,y hat, and J. Then compute the derivatives of the features alongside z and j. Finally update both J and features using grad descent.

# vectorization:
This is a method in which we create vectors of features and training examples and then multiply them together using the following code:

z = np.dot(w,x) + b and this will provide all the values added together and finally pushed into the variable z. It is a much faster way of doing calculations instead of using a for loop. similarly we can also exponentiate vector values to another vector using :
vector_2 = np.exp(vector_1). This also is much faster than the usual for loop.

# vectorization of logistic regression:

This is done by using X(n_x,m) matrix for the training examples, b row (1,m) matrix for the weights, wT(1,n_x) row matrix for the parameters and Z(1,m) matrix for the sigmoid input, then we have the y hat (1,m) row matrix for the sigmoid output.

This prevents the usage of explicit for loops that slow down the network substantially.
The formula is as follows: Z = np.dot(w.T, X) + b, where b is any real number; during calculations python will automatically expand the real number into a matrix of same dimensions as the operand. This is called broadcasting and is very useful in a lot of cases.

## useful numpy commands: 
np.sum(vector_name) can be used to find the sum of all the elements present in the matrix.
np.linalg.norm(vector_name, axis = 0 or 1 | None, keepdims = False | True) helps to normalize the vector, this speeds-up the process of gradient descent.

# Final algorithm:

1. Z = np.dot(w.T, X) + b
2. A = sigmoid(Z)
3. dZ = A - Y where Y is all the output values that were provided in the dataset. Its a matrix.
4. dw = (1/m) * np.dot(X,dZ.T)
5. db = (1/m) * np.sum(dZ) 
This will do both forward propagation and back-propagation without the use of for loops, hence it is much faster (around 300 times) than the for loop implementation.
finally the grad descent:
6. w = w - (learning rate) * dw
7. b = b - (learning rate) * db