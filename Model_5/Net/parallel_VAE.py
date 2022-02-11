#from .Utils_imp_VAE import *
#from .general_utils import conv_output_shape
import sys
import os

sys.path.append(os.path.join("..","DL_utils"))

from B_VAE.Utils_imp_VAE import *
from B_VAE.general_utils import conv_output_shape

from torch import nn
import torch
import numpy as np

class b_encodeco(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 repr_sizes=[32,64,128,256],
                 layer_sizes=[300],
                 latent_space_size=20,
                 conv_kernel_size=5,
                 activators=[nn.Tanh(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=True,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=1,
                 in_device="cpu"
                ):
        super(b_encodeco,self).__init__()

        self.losses={}
        

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size
        self.activators=activators

        self.layer_sizes=layer_sizes
        self.NN_input=(self.compute_odim(image_dim,repr_sizes)[0]*self.compute_odim(image_dim,repr_sizes)[1])*repr_sizes[-1]
        self.latent_space_size=latent_space_size
        self.in_device=in_device
        
        #To GPU 1
        self.encoder_conv=(b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )).to('cuda:0')

        #To GPU 2
        self.encoder_NN_mu=(NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )).to('cuda:1')

        self.encoder_NN_sig=(NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )).to('cuda:1')
        
        self.flatten=s_view()
        #To GPU 3
        self.decoder_NN=(NeuralNet(self.latent_space_size,
                                        self.NN_input,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )).to('cuda:2')

        #To GPU 4
        self.decoder_conv=(b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )).to('cuda:3')
        self.lact=(nn.Sigmoid()).to('cuda:3')
        
    def compute_odim(self,idim,repr_sizes):
        if isinstance(self.conv_pooling,bool):
            pool_l=[self.conv_pooling for i in range(len(repr_sizes))]
        else:
            pool_l=self.conv_pooling

        odim=idim
        for i in range(len(repr_sizes)+np.sum(np.array(pool_l).astype(int))):
            odim=conv_output_shape(odim,kernel_size=self.conv_kernel_size, stride=1, pad=0, dilation=1)
        return odim

    def reparametrization(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        
        esp=torch.randn(*mu.size()).to('cuda:1')
        z=mu+std*esp
        return z
        
    
    def forward(self,x):
        x=self.encoder_conv(x.to(self.in_device))
        x=self.flatten(x.to('cpu'))
        #FCNN
        mu=self.encoder_NN_mu(x.to('cuda:1'))
        sig=self.encoder_NN_sig(x.to('cuda:1'))
        
        z=self.reparametrization(mu,sig)
        z=self.decoder_NN(z.to('cuda:2'))
        z=self.flatten(z.to('cpu'))
        z=self.decoder_conv(z.to('cuda:3'))
        z=self.lact(z.to('cuda:0'))
        
        return z,mu.to('cuda:0'),sig.to('cuda:0')

    def reconstruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE

    def KLD_loss(self,z_mean,z_logvar):
        KLD=-0.5*torch.mean(1+z_logvar-z_mean.pow(2)-z_logvar.exp())
        return KLD

    def ELBO(self,x):
        x_r,z_mean,z_logvar=self.forward(x)
        reconstruction=self.reconstruction_loss(x_r,x)
        KLD=self.KLD_loss(z_mean,z_logvar)
        loss=reconstruction\
            +KLD
        #BUILD LOSSES DICT
        self.losses['KLD']=KLD
        self.losses['reconstruction']=reconstruction
        self.losses["total_loss"]=loss
        
        return self.losses