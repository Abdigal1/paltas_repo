from matplotlib.cbook import ls_mapper
from .Utils_imp_VAE import *
from torch import nn
import torch
from .general_utils import conv_output_shape
import numpy as np

class GMVAE(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 repr_sizes=[32,64,128,256],
                 layer_sizes=[300],
                 w_latent_space_size=20,
                 z_latent_space_size=20,
                 y_latent_space_size=20,
                 conv_kernel_size=5,
                 activators=[nn.Tanh(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=True,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=1,
                 device="cpu"
                ):
        super(GMVAE,self).__init__()
        

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size
        self.activators=activators

        self.layer_sizes=layer_sizes
        self.NN_input=(self.compute_odim(image_dim,repr_sizes)[0]*self.compute_odim(image_dim,repr_sizes)[1])*repr_sizes[-1]
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        self.P=P_NET(input=self.NN_input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
                    )

        self.flatten=s_view()

        self.Q=Q_NET(input=self.NN_input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
                    )

        self.decoder_conv=b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )
        
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
        py=self.Q.py_wz_sig(torch.cat((qw,qz),dim=1))

        
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
        
        return x_recon,qw_x_mu,qw_x_logsig,qz_x_mu,qz_x_logsig,py

    def reconstruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE

    def conditional_prior(self,z_x,z_x_mean,z_x_logvar,y_wz,z_wy,z_wy_mean,z_wy_logvar):
        #inferences
        #TODO: self.particles
        #z_x,z_x_mean,z_x_logvar=self.Q.z_infer(x)
        #w_x,w_x_mean,w_x_logvar=self.Q.w_infer(x)
        #y_wz=self.Q.y_gener(w_x,z_x) #[batch,K]
        #generation
        #z_wy,z_wy_mean,z_wy_logvar=self.P.z_gener(w_x) #[batch,K,z_dim]

        z_x_var=z_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        logq=-0.5*torch.mean(z_x_logvar)-0.5*torch.mean((z_x-z_x_mean)**2/z_x_var)
        
        
        z_wy_var=z_wy_logvar.mul(0.5).exp_() #[batch,K,z_dim]
        log_det_sig=torch.mean(z_wy_logvar,dim=2) #[batch,K]
        MSE=torch.mean((z_wy-z_wy_mean)**2/(z_wy_var**2),dim=2) #[batch,K]
        logp=-0.5*log_det_sig-0.5*MSE #[batch,K]
        yplogp=torch.mean(logp.mul(y_wz)) #[batch,K]
        cond_prior=logq-yplogp
        #TODO: check reduction
        return cond_prior

    def w_prior(self,w_x_mean,w_x_logvar):
        #inferences
        #w_x,w_x_mean,w_x_logvar=self.Q.w_infer(x)

        w_x_var=w_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        KL_w=0.5*torch.mean(w_x_var+w_x_mean**2-1-w_x_logvar)
        #TODO: check reduction
        return KL_w

    def y_prior(self,py):
        #inferences
        #py=self.Q.y_gener(w,z)

        y_prior=torch.mean(torch.sum(-py*(np.log(self.y_latent_space_size,dtype="float32")+torch.log(py))))
        return y_prior

    def ELBO(self,x_i):
        #CNN encoding
        x=self.encoder_conv(x_i)
        x=self.flatten(x)

        #inference
        z_x,z_x_mean,z_x_logvar=self.Q.z_infer(x)
        w_x,w_x_mean,w_x_logvar=self.Q.w_infer(x)
        py_wz=self.Q.y_gener(w_x,z_x) #[batch,K]
        #Generation
        z_wy,z_wy_mean,z_wy_logvar=self.P.z_gener(w_x) #[batch,K,z_dim]
        x_mean=self.P.x_gener(z_x)

        #CNN decoding
        x_mean=self.flatten(x_mean)
        x_mean=self.decoder_conv(x_mean)

        #TODO: check signos
        loss=self.reconstruction_loss(x_mean,x_i)\
            -self.conditional_prior(z_x,z_x_mean,z_x_logvar,py_wz,z_wy,z_wy_mean,z_wy_logvar)\
            -self.w_prior(w_x_mean,w_x_logvar)\
            -self.y_prior(py_wz)
        return loss