import os
import sys

sys.path.append(os.path.join("..",".."))
sys.path.append(os.path.join("..","..","..",'Data_prep'))
#sys.path.append(os.path.join(".."))

import numpy as np

import pickle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def save_generation(fl_u,fl_sig,idx,batch,out_meta_dir=""):
    latent_metadata={}
    latent_metadata['u']=fl_u
    latent_metadata['sig']=fl_sig
    latent_metadata['Place']=batch['Place'][idx]
    latent_metadata['Date']=batch['Date'][idx]
    latent_metadata['Class']=batch['landmarks'][idx]
    file=open(os.path.join(out_meta_dir,(batch['Date'][idx]+'_'+batch['Place'][idx]+'.pkl')),'wb')
    pickle.dump(latent_metadata,file)
    file.close()
    
v_save_generation=np.vectorize(save_generation,signature="(a),(b),(),(),()->()")

def read_results(npy,fold):
    train=npy.tolist()[fold]['train']
    test=npy.tolist()[fold]['valid']
    return train,test

def gen_metadata_from_model(data_base,out_meta_dir,model):
    model.eval()
    dataloader_eval=torch.utils.data.DataLoader(data_base,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             drop_last=False)
    for idx, batch in tqdm(enumerate(dataloader_eval),desc="latent_vars"):
        latent_metadata={}
        model.to('cpu')
        fl=model.encoder_conv(batch['PhantomRGB'])
        fl_=model.flatten(fl)
        fl_u=model.encoder_NN_mu(fl_)
        fl_sig=model.encoder_NN_sig(fl_)
        latent_metadata['u']=fl_u.detach().numpy()
        latent_metadata['sig']=fl_sig.detach().numpy()
        latent_metadata['Place']=batch['Place']
        latent_metadata['Date']=batch['Date']
        latent_metadata['Class']=batch['landmarks']

        file=open(os.path.join(out_meta_dir,(batch['Date'][0]+'_'+batch['Place'][0]+'.pkl')),'wb')
        pickle.dump(latent_metadata,file)
        file.close()


        tqdm.write(
                "Place {loss}\tDate {bce}\tClass {kld}".format(
                    loss=batch['Place'][0],
                    bce=batch['Date'][0],
                    kld=batch['landmarks'][0]
            )
            )

def parallel_gen_metadata_from_VAE(data_base,out_meta_dir,model,batch_size,num_workers,args=['PhantomRGB'],device_in='cuda'):
    #Check directory
    if len(os.listdir(out_meta_dir))>0:
        print("error carpeta ocupada")
        raise (InterruptExecution('ocupied directory'))
    model.eval()
    dataloader_eval=torch.utils.data.DataLoader(data_base,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             drop_last=False)
    for idx, batch in tqdm(enumerate(dataloader_eval),desc="latent_vars"):
        iargs=(batch[arg].to(device_in) for arg in args)
        fl=model.encoder_conv(*(iargs))
        fl_=model.flatten(fl)
        fl_u=model.encoder_NN_mu(fl_)
        fl_sig=model.encoder_NN_sig(fl_)

        #Batch,Size
        v_save_generation(fl_u.cpu().detach().numpy(),
                  fl_sig.cpu().detach().numpy(),
                  np.arange(0,fl_sig.cpu().detach().numpy().shape[0]),#asing batch_size
                  batch,out_meta_dir)


        tqdm.write(
                "Place {loss}\tDate {bce}\tClass {kld}".format(
                    loss=batch['Place'][0],
                    bce=batch['Date'][0],
                    kld=batch['landmarks'][0]
            )
            )