# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Loss functions used for training our model

"""

import tensorflow as tf
import numpy as np

class LossFunctions:
    eps = 0.

    def binary_cross_entropy(self, real, logits, average=True):
      """Binary Cross Entropy between the true and predicted outputs
          loss = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

      Args:
          real: (array) corresponding array containing the true labels
          logits: (array) corresponding array containing the output logits
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if self.eps > 0.0:
        max_val = np.log(1.0 - self.eps) - np.log(self.eps)
        logits = tf.clip_by_value(logits, -max_val, max_val)
      loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=real), axis=-1)
      if average:
        return tf.reduce_mean(loss)
      else:
        return loss
      

    def mean_squared_error(self, real, predictions, average=True):
      """Mean Squared Error between the true and predicted outputs
         loss = (1/n)*Σ(real - predicted)^2

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = tf.square(real - predictions)
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)    


    def kl_gaussian(self, mean, logVar, average=True):
      """KL Divergence between the posterior and a prior gaussian distribution (N(0,1))
         loss = (1/n) * -0.5 * Σ(1 + log(σ^2) - σ^2 - μ^2)

      Args:
          mean: (array) corresponding array containing the mean of our inference model
          logVar: (array) corresponding array containing the log(variance) of our inference model
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = -0.5 * tf.reduce_sum(1 + logVar - tf.exp(logVar) - tf.square(mean + self.eps), 1 ) 
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)


    def kl_categorical(self, qx, log_qx, k, average=True):
      """KL Divergence between the posterior and a prior uniform distribution (U(0,1))
         loss = (1/n) * Σ(qx * log(qx/px)), because we use a uniform prior px = 1/k 
         loss = (1/n) * Σ(qx * (log(qx) - log(1/k)))

      Args:
          qx: (array) corresponding array containing the probs of our inference model
          log_qx: (array) corresponding array containing the log(probs) of our inference model
          k: (int) number of classes
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = tf.reduce_sum(qx * (log_qx - tf.log(1.0/k)), 1)
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)
    
    
    def log_normal(self, x, mu, var):
      """Logarithm of normal distribution with mean=mu and variance=var
         log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

      Args:
         x: (array) corresponding array containing the input
         mu: (array) corresponding array containing the mean 
         var: (array) corresponding array containing the variance

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if self.eps > 0.0:
        var = var + self.eps
      return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis=-1)
    
    
    def labeled_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior, average=True):
      """Variational loss when using labeled data without considering reconstruction loss
         loss = log q(z|x,y) - log p(z) - log p(y)

      Args:
         z: (array) array containing the gaussian latent variable
         z_mu: (array) array containing the mean of the inference model
         z_var: (array) array containing the variance of the inference model
         z_mu_prior: (array) array containing the prior mean of the generative model
         z_var_prior: (array) array containing the prior variance of the generative mode
         average: (bool) whether to average the result to obtain a value
         
      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior) 
      loss = loss - np.log(0.1)
      if average:
        return tf.reduce_mean(loss)
      else:
        return loss
    
    
    def entropy(self, logits, targets, average=True):
      """Entropy loss
          loss = (1/n) * -Σ targets*log(predicted)

      Args:
          logits: (array) corresponding array containing the logits of the categorical variable
          real: (array) corresponding array containing the true labels
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      log_q = tf.nn.log_softmax(logits)
      if average:
        return -tf.reduce_mean(tf.reduce_sum(targets * log_q, 1))
      else:
        return -tf.reduce_sum(targets * log_q, 1)

