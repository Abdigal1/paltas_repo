from .Utils_VAE import *
from torch import nn
import torch
from .general_utils import conv_output_shape

class b_encodeco(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 repr_sizes=[32,64,128,256],
                 layer_sizes=[300],
                 latent_space_size=20,
                 device="cpu"
                ):
        super(b_encodeco,self).__init__()
        
        self.layer_sizes=layer_sizes
        self.NN_input=(self.compute_odim(image_dim,repr_sizes)[0]*self.compute_odim(image_dim,repr_sizes)[1])*repr_sizes[-1]
        self.latent_space_size=latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,repr_sizes=repr_sizes)
        self.encoder_NN_mu=NeuralNet(self.NN_input,self.latent_space_size,layer_sizes=self.layer_sizes)
        self.encoder_NN_sig=NeuralNet(self.NN_input,self.latent_space_size,layer_sizes=self.layer_sizes)
        
        self.flatten=s_view()
        
        self.decoder_NN=NeuralNet(self.latent_space_size,self.NN_input,layer_sizes=self.layer_sizes[::-1])
        self.decoder_conv=b_decoder_conv(image_channels=image_channels,repr_sizes=repr_sizes)
        self.lact=nn.Sigmoid()
        
    def compute_odim(self,idim,repr_sizes):
        odim=idim
        for i in repr_sizes:
            odim=conv_output_shape(odim,kernel_size=3, stride=2, pad=1, dilation=1)
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
        z=self.lact(z)
        
        return z,mu,sig