from matplotlib.cbook import ls_mapper
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
                 w_latent_space_size=20,
                 z_latent_space_size=20,
                 y_latent_space_size=20,
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
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        self.P=P_NET(input=self.input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size
                    )

        self.flatten=s_view()

        self.Q=Q_NET(input=self.input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size
                    )

        self.decoder_conv=b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
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
        #ENCODER
        x=self.encoder_conv(x)
        x=self.flatten(x)
        #FCNN
        #Q(z|x)
        qz_x_mu=self.Q.qz_x_mu(x)
        qz_x_logsig=self.Q.qz_x_sig(x)
        qz=self.Q.reparametrization(qz_x_mu,qz_x_logsig)
        #Q(w|x)
        qw_x_mu=self.Q.qw_x_mu(x)
        qw_x_logsig=self.Q.qw_x_sig(x)
        qw=self.Q.reparametrization(qw_x_mu,qw_x_logsig)
        #P(y|w,z)
        print("debug")
        print(qw.shape)
        print(qz.shape)
        print(torch.cat((qw,qz),dim=1).shape)
        y_logsig=self.Q.py_wz_sig(torch.cat((qw,qz),dim=1))

        
        #z=self.reparametrization(mu,sig)
        #DECODER
        #P(z|w,y)
        #opcional
        #pz_mu=self.P.pz_wy_mu(qw)
        #pz_logsig=self.P.pz_wy_sig(qw)
        #P(x|z)
        x_recon=self.P.px_z(qz)
        x_recon=self.reparametrization(x_recon,qw_x_logsig)

        x_recon=self.flatten(x_recon)
        x_recon=self.decoder_conv(x_recon)
        x_recon=self.lact(x_recon)
        
        return x_recon,qw_x_mu,qw_x_logsig,qz_x_mu,qz_x_logsig,y_logsig

    def recontruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE

    def conditional_prior(self,x):
        z_x,z_x_mean,z_x_logvar=self.Q.z_infer(x)
        z_x_var=z_x_logvar.mul(0.5).exp_()
        logq=-0.5*torch.sum(z_x_logvar)-0.5*torch.sum((z_x-z_x_mean)**2/z_x_var)

    #def w_prior(self,):

    #def y_prior(self,):

    #def L2_loss(self,):

    #def loss_fn_b(r_x,x,mu,sig):
        #BCE=F.binary_cross_entropy(r_x,x,reduction='mean')
        #KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())
        #return BCE+KLD,BCE,KLD

    #def ELBO():