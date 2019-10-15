"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""

import tensorflow as tf

class Networks:
    eps = 1e-6
    
    def __init__(self, params):
      if params is not None:
        self.temperature = tf.placeholder(tf.float32, [])
        self.gaussian_size = params.gaussian_size
        self.hard_gumbel = params.hard_gumbel
        self.loss_type = params.loss_type
        self.dataset = params.dataset


    def latent_gaussian(self, hidden, gaussian_size):
      """Sample from the Gaussian distribution
      
      Args:
        hidden: (array) [batch_size, n_features] features obtained by the encoder
        gaussian_size: (int) size of the gaussian sample vector
        
      Returns:
        (dict) contains the nodes of the mean, log of variance and gaussian
      """
      out = hidden
      mean = tf.layers.dense(out, units=gaussian_size)
      var = tf.layers.dense(out, units=gaussian_size, activation=tf.nn.softplus)
      noise = tf.random_normal(tf.shape(mean), mean = 0, stddev = 1, dtype= tf.float32)
      z = mean + tf.sqrt(var + self.eps) * noise
      return {'mean': mean, 'var': var, 'gaussian': z}
    

    def sample_gumbel(self, shape):
      """Sample from Gumbel(0, 1)
      
      Args:
         shape: (array) containing the dimensions of the specified sample
      """
      U = tf.random_uniform(shape, minval=0, maxval=1)
      return -tf.log(-tf.log(U + self.eps) + self.eps)


    def gumbel_softmax(self, logits, temperature, hard=False):
      """Sample from the Gumbel-Softmax distribution and optionally discretize.
      
      Args:
        logits: (array) [batch_size, n_class] unnormalized log-probs
        temperature: (float) non-negative scalar
        hard: (boolean) if True, take argmax, but differentiate w.r.t. soft sample y
        
      Returns:
        y: (array) [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
      gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
      y = tf.nn.softmax(gumbel_softmax_sample / self.temperature)
      if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
      return y

    
    def encoder_fc(self, input_data, num_classes, is_training=False):
      """Fully connected inference network
      
      Args:
        input_data: (array) [batch_size, n_features=784] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      #reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder_fc')) > 0
      with tf.variable_scope('encoder_fc', reuse=not is_training):
        out = input_data
        
        # encoding from input image to deterministic features
        out = tf.layers.dense(out, units=512)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=512)
        out = tf.nn.relu(out)

        # defining layers to learn the categorical distribution
        logits = tf.layers.dense(out, units=num_classes)
        categorical = self.gumbel_softmax(logits, self.temperature, self.hard_gumbel)
        prob = tf.nn.softmax(logits)
        log_prob = tf.log(prob + self.eps)
        
        # defining layers to learn the gaussian distribution
        concat = tf.concat([categorical, input_data], 1)
        concat = tf.layers.dense(concat, units=512)
        concat = tf.nn.relu(concat)
        concat = tf.layers.dense(concat, units=512)
        concat = tf.nn.relu(concat)
        gaussian = self.latent_gaussian(concat, self.gaussian_size)
        
        # keep graph output operations that will be used in loss functions
        output = gaussian
        output['categorical'] = categorical
        output['prob_cat'] = prob
        output['log_prob_cat'] = log_prob
        output['features'] = gaussian['mean']
        output['logits'] = logits
      return output
    
    
    def gaussian_from_categorical(self, categorical, is_training=False):
      """Fully connected categorical to gaussian p(z|y)
      
      Args:
        categorical: (array) [batch_size, num_classes] latent categorical vector
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) arrays containing the mean and variance
      """
      with tf.variable_scope('decoder_cat', reuse=not is_training):        
        # defining layers to generate a gaussian distribution per category
        mean = tf.layers.dense(categorical, units=self.gaussian_size)
        var = tf.layers.dense(categorical, units=self.gaussian_size, activation=tf.nn.softplus)
      return mean, var
    
    def output_from_gaussian(self, gaussian, output_size, is_training=False):    
      """Fully connected gaussian to output p(x|z)
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the output logits and generated/reconstructed image
      """
      with tf.variable_scope('decoder_gauss', reuse=not is_training):                
        # define layers to generate output given a gaussian variable
        out = tf.layers.dense(gaussian, units=512)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=512)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=output_size)
        if self.loss_type == 'bce':
          reconstructed = tf.nn.sigmoid(out)
        else:
          reconstructed = out
      return out, reconstructed
    
    
    def decoder_fc(self, gaussian, categorical, output_size, is_training=False):
      """Fully connected generative network
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      #reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='decoder_fc')) > 0
      mean, var = self.gaussian_from_categorical(categorical, is_training)        
      out, reconstructed = self.output_from_gaussian(gaussian, output_size, is_training)
        
      return out, mean, var, reconstructed


    def encoder(self, input_data, num_classes, is_training=False):
      """Inference/Encoder network
      
      Args:
        input_data: (array) [batch_size, n_features] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      latent_spec = self.encoder_fc(input_data, num_classes, is_training)
      return latent_spec
    
    
    def decoder(self, gaussian, categorical, output_size, is_training=False):
      """Generative/Decoder network of our model
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      output = self.decoder_fc(gaussian, categorical, output_size, is_training)
      return output
