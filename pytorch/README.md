# Gaussian Mixture Variational Autoencoder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jGOAgwleppSMtUsr7XaldRNBbiwBMhxd)

Implementation of Gaussian Mixture Variational Autoencoder (GMVAE) for Unsupervised Clustering in tensorflow. The model is based on the M2 Unsupervised model proposed by [Kingma et al.](https://arxiv.org/pdf/1406.5298) for semi-supervised learning. Unlike other implementations that use marginalization for the categorical latent variable, we use the [Gumbel-Softmax distribution](https://arxiv.org/pdf/1611.01144), resulting in better time complexity because of the reduced number of gradient estimations. We modified the M2 generative model to represent a Mixture of Gaussians.

### Dependencies

1. [PyTorch](https://pytorch.org/). We tested our method with the **1.3.0** pytorch version. You can Install PyTorch by following the instructions on its website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

2. [Python 3.6.8](https://www.python.org/downloads/). We implemented our method with the **3.6.8** version. Additional libraries include: numpy, scipy and matplotlib.
