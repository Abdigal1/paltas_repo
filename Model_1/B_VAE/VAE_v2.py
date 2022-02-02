from .Utils_imp_VAE import *
from torch import nn
import torch
from .general_utils import conv_output_shape
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
                 device="cpu"
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
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        self.encoder_NN_mu=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.encoder_NN_sig=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )
        
        self.flatten=s_view()
        
        self.decoder_NN=NeuralNet(self.latent_space_size,
                                        self.NN_input,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )

        self.decoder_conv=b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )
        #self.lact=nn.Sigmoid()
        
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
        
        esp=torch.randn(*mu.size()).to(self.device)
        z=mu+std*esp
        return z
        
    
    def forward(self,x):
        x=self.encoder_conv(x)
        x=self.flatten(x)
        #FCNN
        mu=self.encoder_NN_mu(x)
        sig=self.encoder_NN_sig(x)
        
        z=self.reparametrization(mu,sig)
        z=self.decoder_NN(z)
        z=self.flatten(z)
        z=self.decoder_conv(z)
        #z=self.lact(z)
        
        return z,mu,sig

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

