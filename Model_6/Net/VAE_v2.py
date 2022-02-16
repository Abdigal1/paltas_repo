#from .Utils_imp_VAE import *
#from .general_utils import conv_output_shape
import sys
import os

sys.path.append(os.path.join("..","DL_utils"))

from B_VAE.Utils_imp_VAE import *
from B_VAE.general_utils import conv_output_shape

from .P_NET import P_NET
from .Q_NET import Q_NET

from torch import nn
import torch
import numpy as np

class GMVAE(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 non_uniform_dim=0,
                 repr_sizes=[32,64,128,256],
                 pre_layer_sizes=[300],
                 layer_sizes=[300],
                 pre_output=20,
                 w_latent_space_size=20,
                 z_latent_space_size=20,
                 y_latent_space_size=20,
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
        super(GMVAE,self).__init__()

        self.parallelized=Multi_GPU
        self.in_device=in_device
        self.losses={}

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size
        self.activators=activators

        self.non_uniform_input_dim=non_uniform_dim
        self.NN_input=(self.compute_odim(image_dim,repr_sizes,stride=stride)[0]*self.compute_odim(image_dim,repr_sizes,stride=stride)[1])*repr_sizes[-1]
        self.pre_output=pre_output
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size

        self.pre_layer_sizes=pre_layer_sizes
        self.layer_sizes=layer_sizes

        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        self.pre_encoder=NeuralNet(self.NN_input+self.non_uniform_input_dim,
                                    self.pre_output,
                                    layer_sizes=self.pre_layer_sizes,
                                    batch_norm=self.NN_batch_norm
                                    )

        self.Q=Q_NET(input=self.pre_output,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
                    )

        self.flatten=s_view()

        self.P=P_NET(input=self.pre_output,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
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
            self.Q.to('cuda:1')
            self.flatten.to('cpu')
            self.P.to('cuda:2')
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
        
        return x_recon,qw_x_mu,qw_x_logsig,qz_x_mu,qz_x_logsig,py

    def reconstruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE

    def conditional_prior(self,z_x,z_x_mean,z_x_logvar,y_wz,z_wy,z_wy_mean,z_wy_logvar):
        #TODO: self.particles

        z_x_var=z_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        logq=-0.5*torch.mean(z_x_logvar)-0.5*torch.mean((z_x-z_x_mean)**2/(z_x_var**2))
        
        
        z_wy_var=z_wy_logvar.mul(0.5).exp_() #[batch,K,z_dim]
        log_det_sig=torch.mean(z_wy_logvar,dim=2) #[batch,K]
        MSE=torch.mean((z_wy-z_wy_mean)**2/(z_wy_var**2),dim=2) #[batch,K]
        logp=-0.5*log_det_sig-0.5*MSE #[batch,K]
        yplogp=torch.mean(logp.mul(y_wz)) #[batch,K]
        #cond_prior=logq-yplogp
        cond_prior=torch.abs(logq-yplogp)
        return cond_prior

    def w_prior(self,w_x_mean,w_x_logvar):
        w_x_var=w_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        KL_w=0.5*torch.mean(w_x_var**2+w_x_mean**2-1-w_x_logvar)
        return KL_w

    def y_prior(self,py):
        y_prior=torch.mean(torch.sum(py * ( np.log(self.y_latent_space_size,dtype="float32") + torch.log(py) )))
        return y_prior

    def forward_recc_d(self,x_i):
        x=self.encoder_conv(x_i)
        x=self.flatten(x)
        #inference
        z_x,_,_=self.Q.z_infer(x)
        #_=self.Q.y_gener(w_x,z_x) #[batch,K]
        #Generation
        #_,_,_=self.P.z_gener(w_x) #[batch,K,z_dim]
        x_mean=self.P.x_gener(z_x)
        #CNN decoding
        x_mean=self.flatten(x_mean)
        x_mean=self.decoder_conv(x_mean)
        return x_mean

    def forward_recc_u(self,x_i):
        x=self.encoder_conv(x_i)
        x=self.flatten(x)
        #inference
        z_x,_,_=self.Q.z_infer(x)
        w_x,_,_=self.Q.w_infer(x)
        py_wz=self.Q.y_gener(w_x,z_x) #[batch,K]
        #Generation
        z_wy,_,_=self.P.z_gener(w_x) #[batch,K,z_dim]
        x_mean=self.P.x_gener(z_wy[:,torch.argmax(py_wz)])
        #CNN decoding
        x_mean=self.flatten(x_mean)
        x_mean=self.decoder_conv(x_mean)
        return x_mean
    
    def ELBO(self,x_i):
        #CNN encoding
        x=self.encoder_conv((x_i.to(self.in_device if self.parallelized else x_i)))
        x=self.flatten((x.to('cpu') if self.parallelized else x))

        #inference
        z_x,z_x_mean,z_x_logvar=self.Q.z_infer((x.to('cuda:1') if self.parallelized else x))
        w_x,w_x_mean,w_x_logvar=self.Q.w_infer((x.to('cuda:1') if self.parallelized else x))
        py_wz=self.Q.y_gener(
                            (w_x.to('cuda:1') if self.parallelized else w_x),
                            (z_x.to('cuda:1') if self.parallelized else z_x)
                            ) #[batch,K]
        #Generation
        z_wy,z_wy_mean,z_wy_logvar=self.P.z_gener((w_x.to('cuda:2') if self.parallelized else w_x)) #[batch,K,z_dim]
        x_mean=self.P.x_gener((z_x.to('cuda:2') if self.parallelized else z_x))

        #CNN decoding
        x_mean=self.flatten((x_mean.to('cpu') if self.parallelized else x_mean))
        x_mean=self.decoder_conv((x_mean.to('cuda:3') if self.parallelized else x_mean))

        reconstruction=self.reconstruction_loss(
            (x_mean.to("cuda:0") if self.parallelized else x_mean),
            (x_i.to("cuda:0") if self.parallelized else x_i)
            )
        conditional_prior=self.conditional_prior(
            (z_x.to("cuda:0") if self.parallelized else z_x),
            (z_x_mean.to("cuda:0") if self.parallelized else z_x_mean),
            (z_x_logvar.to("cuda:0") if self.parallelized else z_x_logvar),
            (py_wz.to("cuda:0") if self.parallelized else py_wz),
            (z_wy.to("cuda:0") if self.parallelized else z_wy),
            (z_wy_mean.to("cuda:0") if self.parallelized else z_wy_mean),
            (z_wy_logvar.to("cuda:0") if self.parallelized else z_wy_logvar)
            )
        w_prior=self.w_prior(
            (w_x_mean.to("cuda:0") if self.parallelized else w_x_mean),
            (w_x_logvar.to("cuda:0") if self.parallelized else w_x_logvar)
            )

        y_prior=self.y_prior(py_wz)
        loss=reconstruction\
            +conditional_prior\
            +w_prior\
            +y_prior
        #BUILD LOSSES DICT
        self.losses['conditional_prior']=conditional_prior
        self.losses['w_prior']=w_prior
        self.losses['y_prior']=y_prior
        self.losses['reconstruction']=reconstruction
        self.losses["total_loss"]=loss
        
        return self.losses