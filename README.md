## Abstract

This repository has implementations & analysis on some popular machine learning methods.

Libraries:

* Numpy, Keras, Tensorflow, PyTorch

This structure follows the assignments for course CS 480/680: Machine Learning @ the University of Waterloo.

Grades received on the 5 assignments and the literature review are all between 95%-100%. Course mark: 100%.

Below are links to Jupyter notebooks for easier viewing. For full code & data, visit the repository [here](https://github.com/sharma0611/MLfromscratch).

## Contents

1) [**K-Nearest Neighbours & Linear Regression**](./1.%20K%20Nearest%20Neighbours%20&%20Linear%20Regression/KNN%20&%20Linear%20Regression.html)

* Implementations & analysis of KNN & linear regression in numpy with cross-validation & tikhonov regularization.

2) [**Logistic Regression & Mixtures of Gaussians**](./2.%20Logistic%20Regression%20&%20Mixture%20of%20Gaussians/Logistic%20Regression%20&%20Mixture%20of%20Gaussians.html)

* Implementations of logistic regression & mixture of gaussians in numpy with cross-validation & weight regularization.
* Analysis on number of parameters, complexity of hypothesis spaces, inductive biases, and computational complexity between the two algorithms.

3) [**Non-Linear Regression**](./3.%20Non-Linear%20Regression/Non-Linear%20Regression.html)

* Implementations of non-linear regression techniques like regularized generalized linear regression, bayesian generalized linear regression, and gaussian process regression. 
* Analysis on complexity and hypothesis spaces of identity, polynomial, and gaussian kernels.

4) [**Convolutional Neural Networks**](./4.%20Convolutional%20Neural%20Networks/Convolution%20Neural%20Networks.html)

* Implementation of densely connected neural nets and convolution neural nets in Keras and Tensorflow. 
* Comparing RELU and sigmoid activation units in convergence, complexity, and in the context of vanishing gradients.
* Comparing RMSProp, Adagrad, and Adam optimizers in terms of convergence and performance on the given dataset.
* Analysis on various architectural choices like filter sizes, strides, and max-pooling layers.

5) **Recurrent Neural Networks & Transformer Networks**

* Implementation of sequence to sequence encoder/decoder recurrent neural networks with & without attention in PyTorch
* Analysis on linear, GRU, and LSTM units in RNN's in the context of preserving information in the hidden state

6) [**3D Action Recognition - Literature Review**](./3D%20Action%20Recognition%20Literature%20Review.pdf)

* I review the state of the art literature for 3D Action Recognition from pre-deep learning approaches in 2013 to modern deep learning techniques in 2019. 
* I cover the architecture and reasoning behind improved dense trajectories, 3D CNN's, two-stream and single-stream CNN's, and channel seperated networks.
* I review the state of the art performance on UCF101, Sports1M, and Kinetics datasets.


## Next Steps

I'd like to investigate tuning methods for deep residual networks. In the 2015 paper, *Deep Residual Learning for Image Recognition* by He et al., the Microsoft Group researchers introduce the concept of residual blocks to increase the depth of convolution neural networks to 152 layers while alleviating the vanishing gradient problem experienced by earlier architectures like VGG & AlexNet. The skip connections they used allowed residuals to skip blocks of layers and travel deeper back to earlier layers of the network and meaningfully change weights. "Highway Networks" are a closely related architecture that use parameteric gates that learn their own weights through gradient descent which control how much of the residual to let through, however, these did not achieve comparable performance to deep residual networks.

I would like to investigate the potential use of residual gates to improve the learning process for deep residual networks. Fully open residual gates, like the ones in ResNet-152, allow residuals to travel freely throughout all parts of the learning process. However, we could expect that in early parts of the training process we would like earlier layers to have many weight updates in order to converge to some effective feature extracting convolution filters. It could that be desirable that in later training epochs, earlier layers should have less weight updates to allow the network to focus on training deeper layers to create hierarchal features on top of the early layer features. We can achieve this by slowly closing off the residual connections during training from the early layers on to the deeper layers. Optimally, this could allow us to reduce redundancies in the network where early layer features are relearned at later layers of the network because the input of a block, x, is always added to the output given to the next block, F(x). This training method introduces ways for us to bias deep residual networks towards hypotheses that build on low level features. In a way, this is similar to simulated annealing optimization as the network can be considered "hot" when all the connections are open but then as we "cool" the network we expect subnetworks to converge to local optima that are like local constraint satisfaction problems induced by the architecture of the network.
