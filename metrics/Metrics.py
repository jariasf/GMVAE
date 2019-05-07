# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Metrics used to evaluate our model

"""

import tensorflow as tf

class Metrics:
  
  # Code taken from the work 
  # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
  def cluster_acc(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
      w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size
  
  def nmi(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    from sklearn.metrics.cluster import normalized_mutual_info_score
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
