from torch import nn
import torch.nn.functional as F
import torch

def loss_fn(r_x,x,mu,sig):
    BCE=F.binary_cross_entropy(r_x,x,size_average=False)
    KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())
    return BCE+KLD,BCE,KLD

class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

class s_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out):
        super(s_conv, self).__init__()
        self.Conv=nn.Conv2d(repr_size_in,repr_size_out,kernel_size=3,stride=2,padding=1)
        self.act=nn.ReLU()
    def forward(self,x):
        return self.act(self.Conv(x))
    
class s_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out):
        super(s_deconv, self).__init__()
        self.Conv=nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=2,stride=2)
        self.act=nn.ReLU()
    def forward(self,x):
        return self.act(self.Conv(x))

class b_encoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256]):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=[3]+repr_sizes
        
        self.im_layers=nn.ModuleList(
            [
                s_conv(repr_in,repr_out)
                for repr_in,repr_out in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:]
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256]):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=[3]+repr_sizes
        self.repr_sizes.reverse()
        
        self.im_layers=nn.ModuleList(
            [
                s_deconv(repr_in,repr_out)
                for repr_in,repr_out in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:]
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size,layer_sizes=[300,150,50]):
        super(NeuralNet,self).__init__()
        self.layer_sizes=[input_size]+layer_sizes+[output_size]
        self.layers=nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_size,out_size),nn.ReLU())
                for in_size,out_size in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x