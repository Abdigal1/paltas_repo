from turtle import forward
from torch import nn
import torch.nn.functional as F
import torch

def loss_fn(r_x,x,mu,sig):
    BCE=F.binary_cross_entropy(r_x,x,size_average=False)
    KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())
    return BCE+KLD,BCE,KLD

#GMVAE LOSS


class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

class set_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(set_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)

        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
                [act]+\
                ([nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
                ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(set_deconv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
            self.out_pad=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
            self.out_pad=1

        self.comp_layer=nn.ModuleList(
            [nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding,output_padding=self.out_pad)]+\
            [act]+\
            ([nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class b_encoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.Relu(),pooling=True,batch_norm=True,stride=1):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        self.activators=activators
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes))]
        else:
            self.activators=activators
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes))]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm
        
        self.im_layers=nn.ModuleList(
            [
                set_conv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm)
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.batch_norm
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.Relu(),pooling=True,batch_norm=True,stride=1):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        self.activators=activators[::-1]
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes))]
        else:
            self.activators=activators

        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes))]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm
        
        self.im_layers=nn.ModuleList(
            [
                set_deconv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm)
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.activators,
                    self.kernels,
                    self.pooling,
                    self.batch_norm
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
#Add batch normalization,dropout
class NN_layer(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True):
        super(NN_layer,self).__init__()
        self.batch_norm=batch_norm
        self.layer=nn.ModuleList(
            [nn.Linear(inp,out)]+([nn.BatchNorm1d(out)] if self.batch_norm else [])+[act]
            )
    def forward(self,x):
        for sl in self.layer:
            x=sl(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size,layer_sizes=[300,150,50],
                activators=nn.ReLU(),batch_norm=True):
        super(NeuralNet,self).__init__()
        self.layer_sizes=[input_size]+layer_sizes+[output_size]
        self.activators=activators

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(layer_sizes)+1)]
        else:
            self.batch_norm=batch_norm

        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(layer_sizes)+1)]
        else:
            self.activators=activators

        self.layers=nn.ModuleList(
            [
                nn.Sequential(NN_layer(in_size,out_size,act,bat_norm))
                for in_size,out_size,act,bat_norm in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.activators,
                    self.batch_norm
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x

class Q_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes):
        super(Q_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes

        #Q(z|x)
        self.qz_x_mu=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qz_x_sig=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        #Q(w|x)
        self.qw_x_mu=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qw_x_sig=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        batch_norm=self.NN_batch_norm
                                        )
        #P(y|w,z)
        #Input w.shape + z.shape
        #output sigmoid
        # Add small constant to avoid tf.log(0)
        #self.log_py_wz = tf.log(1e-10 + self.py_wz)
        self.py_wz_sig=NeuralNet(self.w_latent_space_size+self.z_latent_space_size,
                                        self.y_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.ReLU for i in len(self.layer_sizes)]+[nn.Softmax()],
                                        batch_norm=self.NN_batch_norm
                                        )
    
    def z_infer(self,x,n_particle=1):
        z_mean=self.qz_x_mu(x)
        z_logsig=self.qz_x_sig(x)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def w_infer(self,x,n_particle=1):
        w_mean=self.qw_x_mu(x)
        w_logsig=self.qw_x_sig(x)
        w=self.reparametrization(w_mean,w_logsig,n_particle)
        return w,w_mean,w_logsig

    def y_gener(self,w,z,n_particle=1):
        #z,z_mean,z_logsig=self.z_infer(x,n_particle)
        #w,w_mean,w_logsig=self.w_infer(x,n_particle)
        c_prob=self.py_wz(torch.cat((w,z),dim=1))
        return c_prob

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample
    #def forward(): ------------------------------------------------------------------------------------------------

class P_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes):
        super(P_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes

        #P(z|w,y)
        self.pz_wy_mu=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.K)])

        self.pz_wy_sig=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.K)])
        #P(x|z)
        self.px_z=NeuralNet(self.z_latent_space_size,#Z
                                        self.NN_input,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )
    def z_gener(self,w,n_particle=1):
        z_mean=self.pz_wy_mu(w)
        z_logsig=self.pz_wy_mu(w)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def x_gener(self,z):
        x=self.px_z(z)
        return x

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample
        
    #def forward(): ------------------------------------------------------------------------------------------------