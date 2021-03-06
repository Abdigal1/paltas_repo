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
                 non_uniform_dim=0,
                 repr_sizes=[32,64,128,256],
                 pre_layer_sizes=[300],
                 layer_sizes=[300],
                 pre_output=20,
                 latent_space_size=20,
                 conv_kernel_size=5,
                 activators=[nn.Tanh(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=True,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=1,
                 device="cpu",
                 Multi_GPU=False,
                 in_device="cpu"
                ):
        super(b_encodeco,self).__init__()

        self.parallelized=Multi_GPU
        self.in_device=in_device
        self.losses={}

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size
        self.activators=activators

        self.NN_input=(self.compute_odim(image_dim,repr_sizes,stride=stride)[0]*self.compute_odim(image_dim,repr_sizes,stride=stride)[1])*repr_sizes[-1]
        self.non_uniform_input_dim=non_uniform_dim
        self.pre_layer_sizes=pre_layer_sizes
        self.layer_sizes=layer_sizes

        self.pre_output=pre_output
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
        
        #self.flatten=s_view()

        self.pre_encoder=NeuralNet(self.NN_input+self.non_uniform_input_dim,
                                    self.pre_output,
                                    layer_sizes=self.pre_layer_sizes,
                                    batch_norm=self.NN_batch_norm
                                    )

        self.encoder_NN_mu=NeuralNet(self.pre_output,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.encoder_NN_sig=NeuralNet(self.pre_output,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )
        
        self.flatten=s_view()
        
        self.decoder_NN=NeuralNet(self.latent_space_size,
                                        self.pre_output,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )

        self.post_encoder=NeuralNet(self.pre_output,
                                    self.NN_input+self.non_uniform_input_dim,
                                    layer_sizes=self.pre_layer_sizes[::-1],
                                    batch_norm=self.NN_batch_norm
                                    ) # TODO: Add other input

        self.decoder_conv=b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )
        if self.parallelized:
            self.encoder_conv.to('cuda:0')
            self.pre_encoder.to('cuda:1')
            self.encoder_NN_mu.to('cuda:1')
            self.encoder_NN_sig.to('cuda:1')
            self.flatten.to('cpu')
            self.decoder_NN.to('cuda:2')
            self.post_encoder.to('cuda:2')
            self.decoder_conv.to('cuda:3')

    def compute_odim(self,idim,repr_sizes,stride):
        if isinstance(self.conv_pooling,bool):
            pool_l=[self.conv_pooling for i in range(len(repr_sizes))]
        else:
            pool_l=self.conv_pooling

        odim=idim
        for i in range(len(repr_sizes)+np.sum(np.array(pool_l).astype(int))):
            if stride==1:
                odim=conv_output_shape(odim,kernel_size=self.conv_kernel_size, stride=stride, pad=0, dilation=1)
            elif stride==2:
                odim=conv_output_shape(odim,kernel_size=self.conv_kernel_size, stride=stride, pad=int((self.conv_kernel_size-1)/2), dilation=1)
        return odim

    def reparametrization(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        
        esp=torch.randn(*mu.size()).to(self.device)
        if self.parallelized:
            esp=esp.to('cuda:1')
        z=mu+std*esp
        return z
        
    
    def forward(self,x):
        x=self.encoder_conv((x.to(self.in_device) if self.parallelized else x))
        x=self.flatten((x.to('cpu') if self.parallelized else x))
        #FCNN
        mu=self.encoder_NN_mu((x.to('cuda:1') if self.parallelized else x))
        sig=self.encoder_NN_sig((x.to('cuda:1') if self.parallelized else x))
        
        z=self.reparametrization(mu,sig)
        z=self.decoder_NN((z.to('cuda:2') if self.parallelized else z))
        z=self.flatten((z.to('cpu') if self.parallelized else z))
        z=self.decoder_conv((z.to('cuda:3') if self.parallelized else z))
        #z=self.lact(z)
        
        return z,mu.to('cuda:0'),sig.to('cuda:0')

    def forward_non_uniform(self,x1,x2):
        x1=self.encoder_conv((x1.to("cuda:0") if self.parallelized else x1))
        x1=self.flatten((x1.to("cpu") if self.parallelized else x1))
        #Pre estimation
        x=torch.cat(((x1.to("cpu") if self.parallelized else x1),
        x2.squeeze(1).squeeze(1).to("cpu")),
        dim=1)
        x=self.pre_encoder((x.to("cuda:1") if self.parallelized else x))

        #FCNN
        mu=self.encoder_NN_mu((x.to("cuda:1") if self.parallelized else x))
        sig=self.encoder_NN_sig((x.to("cuda:1") if self.parallelized else x))
        
        #z12=self.reparametrization((mu.to(self.in_device) if self.parallelized else mu),
        #(sig.to(self.in_device) if self.parallelized else sig))
        z12=self.reparametrization((mu),
        (sig))
        #Pre estimation inv
        z12=self.decoder_NN((z12.to("cuda:2") if self.parallelized else z12))

        z12=self.post_encoder((z12.to("cuda:2") if self.parallelized else z12)) #TODO: split image and other input latent variables
        z1,z2=torch.split(z12,[self.NN_input,self.non_uniform_input_dim],dim=1)

        
        z1=self.flatten((z1.to("cpu") if self.parallelized else z1))
        z1=self.decoder_conv((z1.to("cuda:3") if self.parallelized else z1))
        
        return z1,z2,mu,sig


    def reconstruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE
    
    def KLD_loss(self,z_mean,z_logvar):
        KLD=-0.5*torch.mean(1+z_logvar-z_mean.pow(2)-z_logvar.exp())
        return KLD
    
    def ELBO(self,x1,x2):
        x1_r,x2_r,z_mean,z_logvar=self.forward_non_uniform(
            (x1.to('cuda:0') if self.parallelized else x1),
            (x2.to('cuda:0') if self.parallelized else x2))

        reconstruction=self.reconstruction_loss(x1_r.to('cuda:0') if self.parallelized else x1_r,
        x1.to('cuda:0') if self.parallelized else x1)\
            +self.reconstruction_loss(x2_r.to('cuda:0') if self.parallelized else x2_r,
            x2.squeeze(1).squeeze(1).to('cuda:0') if self.parallelized else x2.squeeze(1).squeeze(1))
        KLD=self.KLD_loss(z_mean.to('cuda:0') if self.parallelized else z_mean,
        z_logvar.to('cuda:0') if self.parallelized else z_logvar)

        loss=reconstruction\
            +KLD

        #BUILD LOSSES DICT
        self.losses['KLD']=KLD
        self.losses['reconstruction']=reconstruction
        self.losses["total_loss"]=loss
        
        return self.losses
