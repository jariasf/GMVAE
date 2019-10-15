"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z, 
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid()
    ])

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var
  
  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)
    
    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output


# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
      #if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output
    #return out_inf, out_gen


'''
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    
    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])
    # q(f|x)
    #self.encoder = Encoder(f_dim)
    
    ## p(x|z)
    #self.decoder = Decoder(z_dim)
    # p(x|f)
    #self.decoder = Decoder(f_dim)
    
    # q(z|f,y)
    self.fc1 = nn.Linear(f_dim + c_dim, z_dim)
    self.fc2 = nn.Linear(z_dim, z_dim)
    #self.fc3 = nn.Linear(z_dim, z_dim)
    self.gaussian = Gaussian(z_dim, z_dim)
    
    # q(y|f)
    self.fc4 = nn.Linear(f_dim, f_dim)
    self.fc5 = nn.Linear(f_dim, f_dim)
    self.gumbel = GumbelSoftmax(f_dim, c_dim, n_distributions)
    
    # p(z|y)
    self.y_mu = nn.Linear(c_dim, z_dim)
    self.y_logVar = nn.Linear(c_dim, z_dim)
    
    # p(f|z)
    self.fc6 = nn.Linear(z_dim, f_dim)
    
    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
      #if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 
  
  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1: #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    #for layer in self.inference_qyx:
    #  x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qyx:
      concat = layer(concat)
    return concat

  

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_logVar = F.softplus(self.y_logVar(y))
    return y_mu, y_logVar
  
  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)
    #h1 = F.leaky_relu(self.fc1(concat))
    #h2 = F.leaky_relu(self.fc2(h1))
    h1 = F.relu(self.fc1(concat))
    h2 = F.relu(self.fc2(h1))    
    return self.gaussian(h2)

  # q(y|f)
  def qyx(self, feat, temperature, hard):
    relu_feat = F.relu(feat)
    h1 = F.relu(self.fc4(relu_feat))
    h2 = F.relu(self.fc5(h1))
    
    #relu_feat = F.leaky_relu(feat)
    #h1 = F.leaky_relu(self.fc4(relu_feat))
    #h2 = F.leaky_relu(self.fc5(h1))
    #logits, prob, y = self.gumbel(h2, temperature, hard)
    return self.gumbel(h2, temperature, hard)
  
  def forward(self, x, temperature=1.0, hard=0):
    # q(f|x)
    feat = self.encoder(x)
    #relu_feat = F.leaky_relu(feat)
    
    # q(y|f)
    logits, prob, y = self.qyx(feat, temperature, hard)
    #logits, prob, y = self.gumbel(relu_feat, temperature, hard)
    
    # q(z|f,y)
    mu, logVar, z = self.qzxy(feat, y)
    #mu, logVar, z = self.qzxy(x.view(-1,784), y)
    
    # p(z|y)
    y_mu, y_logVar = self.pzy(y)
    
    # decoder: p(x|z)
    #_feat = F.leaky_relu(self.fc6(z))
    _feat = F.relu(self.fc6(z))
    out = self.decoder(_feat)
    
    #if self.normalized:
    #norm = feat.norm(dim=1, p=2, keepdim=True)
    #feat = feat.div(norm.expand_as(feat))    
    
    #out, out_logits = self.decoder(z, True)
    output = {'features': feat, 'mean': mu, 'var': logVar, 'gaussian':z, #'x_rec_logits': out_logits,
              'logits': logits, 'prob_cat': prob, 'categorical': y,
              'y_mean': y_mu, 'y_var': y_logVar, 'x_rec': out}
    return output
'''
