---
layout: post
title: "Tensorflow to PyTorch: My personal travel guide"
---

When it comes to neural networks and optimization, there are many possible computational frameworks, each of which has its particular design. The two main choices that are of mainstream use today are [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/). This post will present a series of notebooks I created during my switch from TF to PyTorch.

There is currently a "war" between these two frameworks, where we see each one trying to overcome its missing features (Tensorflow 2.0 with eager execution and PyTorch 1.0).
This post will not discuss any of these new features, but instead will trace the two pros and cons at the time of my switch (one year ago):
- **Tensorflow**: efficient and production-oriented, based on a static computation graph
- **PyTorch**: flexible and easy-to-use, dynamic graph, more pythonic.

## Goal
I started using tensorflow some years ago, during a kaggle competition on speech recognition, sponsored by tensorflow org itself. The competition was oriented towards the usage of a deep learning model on constrained devices, such as the Raspberry Pi, for keyword recognition (e.g. "Ok Google"). This specific usage show best the advantages of tensorflow: we could create a binary from a trained network and run it using C++ on the raspberry pi with incredible performances. I'll try to make a blog post on that specific competition soon.

Some years later I started to be more research-oriented, specifically in the field of reinforcement learning (which from the code perspective is a bit chaotic right now).
There are many frameworks, also in RL, which are written in Tensorflow (e.g. [OpenAI Baselines](https://github.com/openai/baselines)), but I gave a look at PyTorch and it seemed very flexible at first (and it was). I understood that I could, for example, directly clip the gradients in DQN (see [this blog post](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) for the Huber-loss/MSE-loss bug).

This series of notebooks is exactly the diary of my personal transition from TF to PyTorch: this is not simply an introduction of concepts and basics of PyTorch, but it does also underline particular differences and similarities between the two.
It is also my personal cheatsheet when I need some "common object" in PyTorch.

## Notebooks
The repo is structured in 5 notebooks:
- [Notebook 1 - Basics](https://github.com/nicomon24/PyTorch101/blob/master/PyTorch101%20-%20Part%201%20-%20Using%20Tensors.ipynb)
  - An introduction to tensors, shapes, randomization, casting
  - Numpy integration
  - Cuda low-level integration
- [Notebook 2 - Gradients](https://github.com/nicomon24/PyTorch101/blob/master/PyTorch101%20-%20Part%202%20-%20Autograd.ipynb)
  - How gradients are computed in PyTorch, how to access them directly
  - Example: simple linear regression
- [Notebook 3 - Neural Networks](https://github.com/nicomon24/PyTorch101/blob/master/PyTorch101%20-%20Part%203%20-%20Neural%20Networks.ipynb)
  - Creation of a NN using high-level components (linear layers, convolutional layer etc..)
  - Example of a loss function and an optimizer
- [Notebook 4 - MNIST example](https://github.com/nicomon24/PyTorch101/blob/master/PyTorch101%20-%20Part%204%20-%20MNIST%20Training.ipynb)
  - Example of a CNN trained on the MNIST dataset
  - PyTorch datasets and data loading
  - Tensorboard integration using tensorboardX
- [Notebook 5 - Remote GPU training](https://github.com/nicomon24/PyTorch101/blob/master/PyTorch101%20-%20Part%205%20-%20Remote%20GPU%20Training.ipynb)
  - Example of a CNN on CIFAR10
  - Training using a CUDA GPU
  - Saving and loading trained models to switch from a remote to local usage

Happy coding!
