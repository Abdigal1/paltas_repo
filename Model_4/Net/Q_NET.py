import sys
import os

sys.path.append(os.path.join("..","DL_utils"))

from B_VAE.Utils_imp_VAE import *
from B_VAE.general_utils import conv_output_shape

from torch import nn
import torch
import numpy as np


class Q_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes,NN_batch_norm=True):
        super(Q_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes
        self.NN_batch_norm=NN_batch_norm

        #Q(z|x)
        self.qz_x_mu=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qz_x_sig=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        )
        #Q(w|x)
        self.qw_x_mu=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qw_x_sig=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        )
        #P(y|w,z)
        #Input w.shape + z.shape
        #output sigmoid
        # Add small constant to avoid tf.log(0)
        #self.log_py_wz = tf.log(1e-10 + self.py_wz)
        self.py_wz=NeuralNet(self.w_latent_space_size+self.z_latent_space_size,
                                        self.y_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Softmax(dim=1)],
                                        batch_norm=self.NN_batch_norm
                                        )
    
    def z_infer(self,x,n_particle=1):
        z_mean=self.qz_x_mu(x)
        z_logsig=self.qz_x_sig(x)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def w_infer(self,x,n_particle=1):
        w_mean=self.qw_x_mu(x)
        w_logsig=self.qw_x_sig(x)
        w=self.reparametrization(w_mean,w_logsig,n_particle)
        return w,w_mean,w_logsig

    def y_gener(self,w,z,n_particle=1):
        #z,z_mean,z_logsig=self.z_infer(x,n_particle)
        #w,w_mean,w_logsig=self.w_infer(x,n_particle)
        py=self.py_wz(torch.cat((w,z),dim=1))
        return py

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        eps=torch.randn_like(mean)
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample
    #def forward(): ------------------------------------------------------------------------------------------------

