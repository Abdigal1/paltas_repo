import sys
import os

sys.path.append(os.path.join("..","DL_utils"))

from B_VAE.Utils_imp_VAE import *
from B_VAE.general_utils import conv_output_shape

from torch import nn
import torch
import numpy as np


class P_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes,NN_batch_norm=True):
        super(P_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes
        self.NN_batch_norm=NN_batch_norm

        #P(z|w,y)
        self.pz_wy_mu=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.y_latent_space_size)])

        self.pz_wy_sig=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.y_latent_space_size)])
        #P(x|z)
        self.px_z=NeuralNet(self.z_latent_space_size,#Z
                                        self.NN_input,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )
    def z_gener(self,w,n_particle=1):
        z_mean=torch.cat([self.pz_wy_mu[i](w).unsqueeze(1) for i in range(self.y_latent_space_size)],dim=1)
        z_logsig=torch.cat([self.pz_wy_sig[i](w).unsqueeze(1) for i in range(self.y_latent_space_size)],dim=1)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def x_gener(self,z):
        x=self.px_z(z)
        return x

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        eps=torch.randn_like(mean)
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample
        
    #def forward(): ------------------------------------------------------------------------------------------------