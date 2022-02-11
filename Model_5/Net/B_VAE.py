from .Utils_VAE import *
from torch import nn
import torch
from .general_utils import conv_output_shape

class b_encodeco(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 repr_sizes=[32,64,128,256],
                 pre_layer_sizes=[300],
                 layer_sizes=[300],
                 pre_output=20,
                 latent_space_size=20,
                 device="cpu"
                ):
        super(b_encodeco,self).__init__()
        
        
        self.NN_input=(self.compute_odim(image_dim,repr_sizes)[0]*self.compute_odim(image_dim,repr_sizes)[1])*repr_sizes[-1]# TODO: add other input dimension
        self.pre_layer_sizes=pre_layer_sizes
        self.layer_sizes=layer_sizes

        self.pre_output=pre_output
        self.latent_space_size=latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,repr_sizes=repr_sizes)

        #self.flatten=s_view()
        self.pre_encoder=NeuralNet(self.NN_input,self.pre_output,layer_sizes=self.pre_layer_sizes) # TODO: Add other input

        self.encoder_NN_mu=NeuralNet(self.pre_output,self.latent_space_size,layer_sizes=self.layer_sizes)
        self.encoder_NN_sig=NeuralNet(self.pre_output,self.latent_space_size,layer_sizes=self.layer_sizes)
        
        self.flatten=s_view()

        self.decoder_NN=NeuralNet(self.latent_space_size,self.pre_output,layer_sizes=self.layer_sizes[::-1])

        self.post_encoder=NeuralNet(self.pre_output,self.NN_input,layer_sizes=self.pre_layer_sizes[::-1]) # TODO: Add other input
        
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
        #Pre estimation
        x=self.pre_encoder(x)

        #FCNN
        mu=self.encoder_NN_mu(x)
        sig=self.encoder_NN_sig(x)
        
        z=self.reparametrization(mu,sig)
        #Pre estimation inv
        z=self.pre_encoder(z)

        z=self.decoder_NN(z)
        z=self.flatten(z)
        z=self.decoder_conv(z)
        z=self.lact(z)
        
        return z,mu,sig

    def forward_non_uniform(self,x1,x2):
        x=self.encoder_conv(x1)
        x=self.flatten(x1)
        #Pre estimation
        print(torch.concat(x1,x2)) #CHECK
        x=self.pre_encoder(torch.concat(x1,x2))

        #FCNN
        mu=self.encoder_NN_mu(x)
        sig=self.encoder_NN_sig(x)
        
        z=self.reparametrization(mu,sig)
        #Pre estimation inv
        z=self.pre_encoder(z) #TODO: split image and other input latent variables
        #z1
        #z2

        z=self.decoder_NN(z)
        z=self.flatten(z)
        z=self.decoder_conv(z)
        z=self.lact(z)
        
        return z,mu,sig