# ML from scratch

This repository has implementations & analysis on popular machine learning methods.

Libraries:

* Numpy, Keras, Tensorflow, PyTorch


## Contents

1) **K-Nearest Neighbours & Linear Regression**

* Implementations & analysis of KNN & linear regression in numpy with cross-validation & tikhonov regularization.

2) **Logistic Regression & Mixtures of Gaussians**

* Implementations of logistic regression & mixture of gaussians in numpy with cross-validation & weight regularization.
* Analysis on number of parameters, complexity of hypothesis spaces, inductive biases, and computational complexity between the two algorithms.

3) **Non-Linear Regression**

* Implementations of non-linear regression techniques like regularized generalized linear regression, bayesian generalized linear regression, and gaussian process regression. 
* Analysis on complexity and hypothesis spaces of identity, polynomial, and gaussian kernels.

4) **Convolutional Neural Networks**

* Implementation of densely connected neural nets and convolution neural nets in Keras and Tensorflow. 
* Comparing RELU and sigmoid activation units in convergence, complexity, and in the context of vanishing gradients.
* Comparing RMSProp, Adagrad, and Adam optimizers in terms of convergence and performance on the given dataset.
* Analysis on various architectural choices like filter sizes, strides, and max-pooling layers.

5) **Recurrent Neural Networks**

* Implementation of sequence to sequence encoder/decoder recurrent neural networks with & without attention in PyTorch
* Analysis on linear, GRU, and LSTM units in RNN's in the context of preserving information in the hidden state

6) **3D Action Recognition - Literature Review**

* I review the state of the art literature for 3D Action Recognition from pre-deep learning approaches in 2013 to modern deep learning techniques in 2019. 
* I cover the architecture and reasoning behind improved dense trajectories, 3D CNN's, two-stream and single-stream CNN's, and channel seperated networks.
* I review the state of the art performance on UCF101, Sports1M, and Kinetics datasets.


This structure follows the assignments for course CS 480/680: Machine Learning @ the University of Waterloo.

My marks on the 5 assignments and the literature review were all between 95%-100%. My course mark was 100%.