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
                 conv_pooling=True,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=1,
                 device="cpu"
                ):
        super(b_encodeco,self).__init__()
        

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size

        self.layer_sizes=layer_sizes
        self.NN_input=(self.compute_odim(image_dim,repr_sizes)[0]*self.compute_odim(image_dim,repr_sizes)[1])*repr_sizes[-1]
        self.latent_space_size=latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        #Q(z|x)
        self.encoder_NN_Qz_x_mu=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )

        self.encoder_NN_Qz_x_sig=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        #Q(w|x)
        self.encoder_NN_Qw_x_mu=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )

        self.encoder_NN_Qw_x_sig=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        #P(y|w,z)
        #Input w.shape + z.shape
        #output sigmoid
        # Add small constant to avoid tf.log(0)
        #self.log_py_wz = tf.log(1e-10 + self.py_wz)

        
        self.flatten=s_view()

        #P(z|w,y)
        self.encoder_NN_Pz_wy_mu=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )

        self.encoder_NN_Pz_wy_sig=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        #P(x|z)
        self.encoder_NN_Px_z=NeuralNet(self.NN_input,
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        
        
        self.lact=nn.Sigmoid()
        
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
        z=self.lact(z)
        
        return z,mu,sig