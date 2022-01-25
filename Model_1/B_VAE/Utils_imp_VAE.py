from torch import nn
import torch.nn.functional as F
import torch

def loss_fn(r_x,x,mu,sig):
    BCE=F.binary_cross_entropy(r_x,x,size_average=False)
    KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())
    return BCE+KLD,BCE,KLD

def loss_fn_b(r_x,x,mu,sig):
    BCE=F.binary_cross_entropy(r_x,x,reduction='mean')
    KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())
    return BCE+KLD,BCE,KLD

def MSEloss_fn_b(r_x,x,mu,sig):
    BCE=F.mse_loss(r_x,x,reduction='mean')
    KLD=-0.5*torch.mean(1+sig-mu.pow(2)-sig.exp())

class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

class set_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,pooling=True,batch_norm=True,stride=1):
        super(set_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)

        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
                [nn.ReLU()]+\
                ([nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
                ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,pooling=True,batch_norm=True,stride=1):
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
            [nn.ReLU()]+\
            ([nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class b_encoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],kernel_size=5,pooling=True,batch_norm=True,stride=1):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        
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
                pooling,
                batch_norm)
                for repr_in,repr_out,kernel_size,pooling,batch_norm in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
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
    
class b_decoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],kernel_size=5,pooling=True,batch_norm=True,stride=1):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        
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
                pooling,
                batch_norm)
                for repr_in,repr_out,kernel_size,pooling,batch_norm in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
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
    def __init__(self,inp,out,batch_norm=True):
        super(NN_layer,self).__init__()
        self.batch_norm=batch_norm
        self.layer=nn.ModuleList(
            [nn.Linear(inp,out)]+([nn.BatchNorm1d(out)] if self.batch_norm else [])+[nn.ReLU()]
            )
    def forward(self,x):
        for sl in self.layer:
            x=sl(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size,layer_sizes=[300,150,50],batch_norm=True):
        super(NeuralNet,self).__init__()
        self.layer_sizes=[input_size]+layer_sizes+[output_size]

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(layer_sizes)+1)]
        else:
            self.batch_norm=batch_norm

        self.layers=nn.ModuleList(
            [
                nn.Sequential(NN_layer(in_size,out_size,bat_norm))
                for in_size,out_size,bat_norm in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.batch_norm
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x