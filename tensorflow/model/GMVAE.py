"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder for Unsupervised Clustering

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from networks.Networks import *
from losses.LossFunctions import *
from metrics.Metrics import *

class GMVAE:

    def __init__(self, params):
      self.batch_size = params.batch_size
      self.batch_size_val = params.batch_size_val
      self.initial_temperature = params.temperature
      self.decay_temperature = params.decay_temperature
      self.num_epochs = params.num_epochs
      self.loss_type = params.loss_type
      self.num_classes = params.num_classes
      self.w_gauss = params.w_gaussian
      self.w_categ = params.w_categorical
      self.w_recon = params.w_reconstruction
      self.decay_temp_rate = params.decay_temp_rate
      self.gaussian_size = params.gaussian_size
      self.min_temperature = params.min_temperature
      self.temperature = params.temperature # current temperature
      self.verbose = params.verbose
      
      self.sess = tf.Session()
      self.network = Networks(params)
      self.losses = LossFunctions()
      
      self.learning_rate = tf.placeholder(tf.float32, [])
      self.lr = params.learning_rate
      self.decay_epoch = params.decay_epoch
      self.lr_decay = params.lr_decay
      
      self.dataset = params.dataset
      self.metrics = Metrics()
      
    
    def create_dataset(self, is_training, data, labels, batch_size):
      """Create dataset given input data

      Args:
          is_training: (bool) whether to use the train or test pipeline.
                       At training, we shuffle the data and have multiple epochs
          data: (array) corresponding array containing the input data
          labels: (array) corresponding array containing the labels of the input data
          batch_size: (int) size of each batch to consider from the data
 
      Returns:
          output: (dict) contains what will be the input of the tensorflow graph
      """
      num_samples = data.shape[0]
      
      # create dataset object      
      if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(data)
      else:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

      # shuffle data in training phase
      if is_training:  
        dataset = dataset.shuffle(num_samples).repeat()

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(1)

      # create reinitializable iterator from dataset
      iterator = dataset.make_initializable_iterator()

      if labels is None:
        data = iterator.get_next()
      else:
        data, labels = iterator.get_next()
   
      iterator_init = iterator.initializer
      output = {'data': data, 'labels': labels, 'iterator_init': iterator_init}
      return output

    
    def unlabeled_loss(self, data, latent_spec, output_size, is_training=True):
      """Model function defining the loss functions derived from the variational lower bound

      Args:
          data: (array) corresponding array containing the input data
          latent_spec: (dict) contains the graph operations or nodes of the latent variables
          output_size: (int) size of the output layer
          is_training: (bool) whether we are in training phase or not

      Returns:
          loss_dic: (dict) contains the values of each loss function and predictions
      """
      gaussian, mean, var  = latent_spec['gaussian'], latent_spec['mean'], latent_spec['var']
      categorical, prob, log_prob = latent_spec['categorical'], latent_spec['prob_cat'], latent_spec['log_prob_cat']
      _logits, features = latent_spec['logits'], latent_spec['features']
      
      output, y_mean, y_var = latent_spec['output'], latent_spec['y_mean'], latent_spec['y_var']

      # reconstruction loss
      if self.loss_type == 'bce':
        loss_rec = self.w_recon * self.losses.binary_cross_entropy(data, output)
      elif self.loss_type == 'mse':
        loss_rec = self.w_recon * tf.losses.mean_squared_error(data, output)
      else:
        raise "invalid loss function... try bce or mse..."
        
      # gaussian loss       
      loss_gaussian = self.w_gauss * self.losses.labeled_loss(gaussian, mean, var, y_mean, y_var)

      # categorical loss
      loss_categorical = self.w_categ * -self.losses.entropy(_logits, prob)

      # obtain predictions
      predicted_labels = tf.argmax(_logits, axis=1)

      # total_loss
      loss_total = loss_rec + loss_gaussian + loss_categorical
      
      loss_dic = {'total': loss_total, 'predicted_labels': predicted_labels,
                  'reconstruction': loss_rec,
                  'gaussian': loss_gaussian,
                  'categorical': loss_categorical}
      return loss_dic
      
    
    def create_model(self, is_training, inputs, output_size):
      """Model function defining the graph operations.

      Args:
          is_training: (bool) whether we are in training phase or not
          inputs: (dict) contains the inputs of the graph (features, labels...)
                  this can be `tf.placeholder` or outputs of `tf.data`
          output_size: (int) size of the output layer

      Returns:
          model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
      """
      data, _labels = inputs['data'], inputs['labels']
      latent_spec = self.network.encoder(data, self.num_classes, is_training)

      out_logits, y_mean, y_var, output = self.network.decoder(latent_spec['gaussian'], 
                                                   latent_spec['categorical'], 
                                                   output_size, is_training)

      latent_spec['output'] = out_logits
      latent_spec['y_mean'] = y_mean
      latent_spec['y_var'] = y_var
      
      # unlabeled losses
      unlabeled_loss_dic = self.unlabeled_loss(data, latent_spec, output_size, is_training)
      
      loss_total = unlabeled_loss_dic['total']
      
      if is_training:
        # use adam for optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # needed for batch normalization layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss_total)      

      # create model specification
      model_spec = inputs
      model_spec['variable_init_op'] = tf.global_variables_initializer()
      
      # optimizers are only available in training phase
      if is_training:
        model_spec['train_op'] = train_op
      else:
        model_spec['output'] = output
      
      model_spec['loss_total'] = loss_total
      model_spec['loss_rec_ul'] = unlabeled_loss_dic['reconstruction']
      model_spec['loss_gauss_ul'] = unlabeled_loss_dic['gaussian']
      model_spec['loss_categ_ul'] = unlabeled_loss_dic['categorical']
      model_spec['true_labels'] = _labels
      model_spec['predicted'] = unlabeled_loss_dic['predicted_labels']
      
      return model_spec    
    

    def evaluate_dataset(self, is_training, num_batches, model_spec):
      """Evaluate the model

      Args:
          is_training: (bool) whether we are training or not
          num_batches: (int) number of batches to train/test
          model_spec: (dict) contains the graph operations or nodes needed for evaluation

      Returns:
          (dic) average of loss functions and metrics for the given number of batches
      """
      avg_accuracy = 0.0    
      avg_nmi = 0.0
      avg_loss_cat = 0.0
      avg_loss_total = 0.0
      avg_loss_rec = 0.0
      avg_loss_gauss = 0.0
      
      list_predicted_labels = []
      list_true_labels = []

      # initialize dataset iteratior
      self.sess.run(model_spec['iterator_init'])
      
      if is_training:
        
        train_optimizer = model_spec['train_op']
        
        # training phase
        for j in range(num_batches):
          _, loss_total, loss_cat_ul, loss_rec_ul, loss_gauss_ul, true_labels, predicted_labels = self.sess.run([train_optimizer,
                                                                             model_spec['loss_total'], model_spec['loss_categ_ul'], 
                                                                             model_spec['loss_rec_ul'], model_spec['loss_gauss_ul'], 
                                                                             model_spec['true_labels'], model_spec['predicted']],
                                                                             feed_dict={self.network.temperature: self.temperature, 
                                                                                        self.learning_rate: self.lr})                                                                                           
          
          # save values
          list_predicted_labels.append(predicted_labels)
          list_true_labels.append(true_labels)
          avg_loss_rec += loss_rec_ul
          avg_loss_gauss += loss_gauss_ul
          avg_loss_cat += loss_cat_ul         
          avg_loss_total += loss_total
      else:
        # validation phase
        for j in range(num_batches):
          # run the tensorflow flow graph
          loss_rec_ul, loss_gauss_ul, loss_cat_ul, loss_total, true_labels, predicted_labels = self.sess.run([ 
                                                     model_spec['loss_rec_ul'], model_spec['loss_gauss_ul'], 
                                                     model_spec['loss_categ_ul'],model_spec['loss_total'], 
                                                     model_spec['true_labels'], model_spec['predicted']],
                                                     feed_dict={self.network.temperature: self.temperature,
                                                                self.learning_rate: self.lr})     

          # save values
          list_predicted_labels.append(predicted_labels)
          list_true_labels.append(true_labels)
          avg_loss_rec += loss_rec_ul
          avg_loss_gauss += loss_gauss_ul
          avg_loss_cat += loss_cat_ul
          avg_loss_total += loss_total

      # average values by the given number of batches 
      avg_loss_rec /= num_batches
      avg_loss_gauss /= num_batches
      avg_loss_cat /= num_batches
      avg_loss_total /= num_batches
      
      # average accuracy and nmi of all the data
      predicted_labels = np.hstack(list_predicted_labels)
      true_labels = np.hstack(list_true_labels)
      avg_nmi = self.metrics.nmi(predicted_labels, true_labels)
      avg_accuracy = self.metrics.cluster_acc(predicted_labels, true_labels)
      
      return {'loss_rec': avg_loss_rec, 'loss_gauss': avg_loss_gauss, 
              'loss_cat': avg_loss_cat, 'loss_total': avg_loss_total, 
              'accuracy': avg_accuracy, 'nmi': avg_nmi}

    
    def train(self, train_data, train_labels, val_data, val_labels):
      """Train the model

      Args:
          train_data: (array) corresponding array containing the training data
          train_labels: (array) corresponding array containing the labels of the training data
          val_data: (array) corresponding array containing the validation data
          val_labels: (array) corresponding array containing the labels of the validation data

      Returns:
          output: (dict) contains the history of train/val loss
      """
      train_history_loss, val_history_loss = [], []
      train_history_acc, val_history_acc = [], []
      train_history_nmi, val_history_nmi = [], []
      
      # create training and validation dataset
      train_dataset = self.create_dataset(True, train_data, train_labels, 
                                          self.batch_size)
      val_dataset = self.create_dataset(False, val_data, val_labels, self.batch_size_val)
      
      self.output_size = train_data.shape[1]
    
      # create train and validation models      
      train_model = self.create_model(True, train_dataset, self.output_size)
      val_model = self.create_model(False, val_dataset, self.output_size)
    
      # set number of batches
      num_train_batches = int(np.ceil(train_data.shape[0] / (1.0 * self.batch_size)))
      num_val_batches = int(np.ceil(val_data.shape[0] / (1.0 * self.batch_size_val)))

      # initialize global variables
      self.sess.run( train_model['variable_init_op'] )

      # training and validation phases
      print('Training phase...')
      for i in range(self.num_epochs):
        
        # decay learning rate according to decay_epoch parameter
        if self.decay_epoch > 0 and (i + 1) % self.decay_epoch == 0:
          self.lr = self.lr * self.lr_decay
          print('Decaying learning rate: %lf' % self.lr)
        
        # evaluate train and validation datasets
        train_loss = self.evaluate_dataset(True, num_train_batches, train_model)
        val_loss = self.evaluate_dataset(False, num_val_batches, val_model)
       
        # get training results for printing
        train_loss_rec = train_loss['loss_rec']
        train_loss_gauss = train_loss['loss_gauss']
        train_loss_cat = train_loss['loss_cat']
        train_accuracy = train_loss['accuracy']
        train_nmi = train_loss['nmi']
        train_total_loss = train_loss['loss_total']
        
        # get validation results for printing
        val_loss_rec = val_loss['loss_rec']
        val_loss_gauss = val_loss['loss_gauss']
        val_loss_cat = val_loss['loss_cat']
        val_accuracy = val_loss['accuracy']
        val_nmi = val_loss['nmi']
        val_total_loss = val_loss['loss_total']        

        # if verbose then print specific information about training
        if self.verbose == 1:
          print("(Epoch %d / %d)" % (i + 1, self.num_epochs) )
          print("Train - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
                (train_loss_rec, train_loss_gauss, train_loss_cat))
          print("Valid - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
                (val_loss_rec, val_loss_gauss, val_loss_cat))
          print("Accuracy=Train: %.5lf; Val: %.5lf   NMI=Train: %.5lf; Val: %.5lf   Total Loss=Train: %.5lf; Val: %.5lf" % \
               (train_accuracy, val_accuracy, train_nmi, val_nmi, train_total_loss, val_total_loss))
        else:
          print("(Epoch %d / %d) Train Loss: %.5lf; Val Loss: %.5lf   Train ACC: %.5lf; Val ACC: %.5lf   Train NMI: %.5lf; Val NMI: %.5lf" % \
                (i + 1, self.num_epochs, train_total_loss, val_total_loss, train_accuracy, val_accuracy, train_nmi, val_nmi))
        
        # save loss and accuracy of each epoch
        train_history_loss.append(train_total_loss)
        val_history_loss.append(val_total_loss)
        train_history_acc.append(train_accuracy)
        val_history_acc.append(val_accuracy)
        
        if self.decay_temperature == 1:
          # decay temperature of gumbel-softmax
          self.temperature = np.maximum(self.initial_temperature*np.exp(-self.decay_temp_rate*(i + 1) ),self.min_temperature)
          if self.verbose == 1:
            print("Gumbel Temperature: %.5lf" % self.temperature)

      return {'train_history_loss' : train_history_loss, 'val_history_loss': val_history_loss,
              'train_history_acc': train_history_acc, 'val_history_acc': val_history_acc}
 
    
    def test(self, test_data, test_labels, batch_size = -1):
      """Test the model with new data

      Args:
          test_data: (array) corresponding array containing the testing data
          test_labels: (array) corresponding array containing the labels of the testing data
          batch_size: (int) batch size used to run the model
          
      Return:
          accuracy for the given test data

      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = test_data.shape[0]
        
      # create dataset
      test_dataset = self.create_dataset(False, test_data, test_labels, batch_size)
      true_labels = test_dataset['labels']
      
      # perform a forward call on the encoder to obtain predicted labels
      latent = self.network.encoder(test_dataset['data'], self.num_classes)
      logits = latent['logits']
      predicted_labels = tf.argmax(logits, axis=1)

      # initialize dataset iterator
      self.sess.run(test_dataset['iterator_init'])
      
      # calculate number of batches given batch size
      num_batches = int(np.ceil(test_data.shape[0] / (1.0 * batch_size)))
      
      # evaluate the model
      list_predicted_labels = []
      list_true_labels = []

      for j in range(num_batches):
        _predicted_labels, _true_labels = self.sess.run([predicted_labels, true_labels], 
                                          feed_dict={self.network.temperature: self.temperature,
                                                     self.learning_rate: self.lr})

        # save values
        list_predicted_labels.append(_predicted_labels)
        list_true_labels.append(_true_labels)

      # average accuracy and nmi of all the data
      predicted_labels = np.hstack(list_predicted_labels)
      true_labels = np.hstack(list_true_labels)
      avg_nmi = self.metrics.nmi(predicted_labels, true_labels)
      avg_accuracy = self.metrics.cluster_acc(predicted_labels, true_labels)
      
      return avg_accuracy, avg_nmi
    

    def latent_features(self, data, batch_size=-1):
      """Obtain latent features learnt by the model

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          features: (array) array containing the features from the data
      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = data.shape[0]
      
      # create dataset  
      dataset = self.create_dataset(False, data, None, batch_size)

      # we will use only the encoder network
      latent = self.network.encoder(dataset['data'], self.num_classes)
      encoder = latent['features']
      
      # obtain the features from the input data
      self.sess.run(dataset['iterator_init'])      
      num_batches = data.shape[0] // batch_size
      
      features = np.zeros((data.shape[0], self.gaussian_size))
      for j in range(num_batches):
        features[j*batch_size:j*batch_size + batch_size] = self.sess.run(encoder,
                                                                        feed_dict={self.network.temperature: self.temperature
                                                                                  ,self.learning_rate: self.lr})
      return features
    
    
    def reconstruct_data(self, data, batch_size=-1):
      """Reconstruct Data

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          reconstructed: (array) array containing the reconstructed data
      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = data.shape[0]
      
      # create dataset
      dataset = self.create_dataset(False, data, None, batch_size)

      # reuse model used in training
      model_spec = self.create_model(False, dataset, data.shape[1])

      # obtain the reconstructed data
      self.sess.run(model_spec['iterator_init'])      
      num_batches = data.shape[0] // batch_size      
      reconstructed = np.zeros(data.shape)
      pos = 0
      for j in range(num_batches):
        reconstructed[pos:pos + batch_size] = self.sess.run(model_spec['output'],
                                                            feed_dict={self.network.temperature: self.temperature
                                                                      ,self.learning_rate:self.lr})
        pos += batch_size
      return reconstructed
    

    def plot_latent_space(self, data, labels, save=False):
      """Plot the latent space learnt by the model

      Args:
          data: (array) corresponding array containing the data
          labels: (array) corresponding array containing the labels
          save: (bool) whether to save the latent space plot

      Returns:
          fig: (figure) plot of the latent space
      """
      # obtain the latent features
      features = self.latent_features(data)
      
      # plot only the first 2 dimensions
      fig = plt.figure(figsize=(8, 6))
      plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
              edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
      plt.colorbar()
      if(save):
          fig.savefig('latent_space.png')
      return fig
    

    def generate_data(self, num_elements=1, category=0):
      """Generate data for a specified category

      Args:
          num_elements: (int) number of elements to generate
          category: (int) category from which we will generate data

      Returns:
          generated data according to num_elements
      """
      indices = (np.ones(num_elements)*category).astype(int).tolist()
      
      # category is specified with a one-hot array
      categorical = tf.one_hot(indices, self.num_classes)
      
      # infer the gaussian distribution according to the category
      mean, var = self.network.gaussian_from_categorical(categorical)
      
      # gaussian random sample by using the mean and variance
      gaussian = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
      
      # generate new samples with the given gaussian
      _, out = self.network.output_from_gaussian(gaussian, self.output_size)
      
      return self.sess.run(out, feed_dict={self.network.temperature: self.temperature
                                          ,self.learning_rate:self.lr})
    

    def random_generation(self, num_elements=1):
      """Random generation for each category

      Args:
          num_elements: (int) number of elements to generate

      Returns:
          generated data according to num_elements
      """
      # categories for each element
      arr = np.array([])
      for i in range(self.num_classes):
        arr = np.hstack([arr,np.ones(num_elements) * i] )
      indices = arr.astype(int).tolist()
      categorical = tf.one_hot(indices, self.num_classes)
      
      # infer the gaussian distribution according to the category
      mean, var = self.network.gaussian_from_categorical(categorical)
      
      # gaussian random sample by using the mean and variance
      gaussian = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
      
      # generate new samples with the given gaussian
      _, out = self.network.output_from_gaussian(gaussian, self.output_size)
      
      return self.sess.run(out, feed_dict={self.network.temperature: self.temperature
                                          ,self.learning_rate:self.lr})
    
