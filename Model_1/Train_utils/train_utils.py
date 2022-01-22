from tqdm import tqdm
import numpy as np
import torch
import os

def train(model,optimizer,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    bce_d=[]
    kld_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="instances"):
        r_img,mu,sig=model(batch["PhantomRGB"].to(device))
        loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tqdm.write(
            "total loss {loss:.4f}\tBCE {bce:.4f}\tKLD {kld:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                bce=bce.item(),
                kld=kld.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )


        #SAVE TRAIN DATA
        loss_d.append(loss.item())
        bce_d.append(bce.item())
        kld_d.append(kld.item())
    return loss_d,bce_d,kld_d


def test(model,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    bce_d=[]
    kld_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="Test"):
        r_img,mu,sig=model(batch["PhantomRGB"].to(device))
        loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        
        tqdm.write(
            "total loss {loss:.4f}\tBCE {bce:.4f}\tKLD {kld:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                bce=bce.item(),
                kld=kld.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )
        
        #SAVE TEST DATA
        loss_d.append(loss.item())
        bce_d.append(bce.item())
        kld_d.append(kld.item())
    return loss_d,bce_d,kld_d

def train_test(model,optimizer,train_set,test_set,batch_size,use_cuda,loss_function,epochs,data_train_dir,in_device=None):
    epoch_loss={}
    epoch_bce={}
    epoch_kld={}

    epoch_loss_train=[]
    epoch_bce_train=[]
    epoch_kld_train=[]

    epoch_loss_test=[]
    epoch_bce_test=[]
    epoch_kld_test=[]
    
    best_result=0

    for epoch in tqdm(range(epochs),desc="Epoch"):

        drop=False
        if len(train_set)%batch_size==1 or len(test_set)%batch_size==1:
            drop=True

        dataloader_train=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=6,drop_last=drop,persistent_workers=True)
        dataloader_test=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=6,drop_last=drop,persistent_workers=True)

        loss_d,bce_d,kld_d=train(model,optimizer,dataloader_train,use_cuda,loss_function,in_device)
    
        epoch_loss_train.append(np.mean(np.array(loss_d)))
        epoch_bce_train.append(np.mean(np.array(bce_d)))
        epoch_kld_train.append(np.mean(np.array(kld_d)))
    
        loss_d,bce_d,kld_d=test(model,dataloader_test,use_cuda,loss_function,in_device)
        #loss_d,bce_d,kld_d=test(model,dataloader_train,use_cuda,loss_function)
        
        if (np.mean(np.array(loss_d)))>best_result:
            best_result=(np.mean(np.array(loss_d)))
            best_model=model.state_dict()
    
        epoch_loss_test.append(np.mean(np.array(loss_d)))
        epoch_bce_test.append(np.mean(np.array(bce_d)))
        epoch_kld_test.append(np.mean(np.array(kld_d)))

        tqdm.write("epoch {epoch:.2f}%".format(
                    epoch=epoch
                    ))

        del dataloader_train
        del dataloader_test

        epoch_loss={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }

        epoch_bce={"train":epoch_bce_train,
                        "valid":epoch_bce_test
                            }

        epoch_kld={"train":epoch_kld_train,
                        "valid":epoch_kld_test
                            }
        

        np.save(os.path.join(data_train_dir,"loss_results"+'.npy'),epoch_loss)
        np.save(os.path.join(data_train_dir,"bce_results"+'.npy'),epoch_bce)
        np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),epoch_kld)

    
    return epoch_loss_train,epoch_bce_train,epoch_kld_train,epoch_loss_test,epoch_bce_test,epoch_kld_test,best_model

def K_fold_train(model,
                dataset,
                epochs,
                batch_size,
                use_cuda,
                folds,
                data_train_dir,
                loss_fn,
                in_device=None
                ):
    fold_loss={}
    fold_bce={}
    fold_kld={}

    #Shuffle data
    train_s=int((len(dataset))*0.7)
    test_s=int(len(dataset)-train_s)
    print("train len")
    print(train_s)
    print("test len")
    print(test_s)    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])

    drop=False
    if train_s%batch_size==1 or test_s%batch_size==1:
        drop=True

    for fold in tqdm(range(folds),desc="folds"):
        train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])

        ed=model
        #optimizer
        optimizer=torch.optim.Adam(ed.parameters(),lr=1e-3)

        #Epochs
        epoch_loss_train,epoch_bce_train,epoch_kld_train,epoch_loss_test,epoch_bce_test,epoch_kld_test,best_model=train_test(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            test_set=test_set,
            batch_size=batch_size,
            use_cuda=use_cuda,
            loss_function=loss_fn,
            epochs=epochs,
            data_train_dir=data_train_dir,
            in_device=in_device
        )

        fold_loss[fold]={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }

        fold_bce[fold]={"train":epoch_bce_train,
                        "valid":epoch_bce_test
                            }

        fold_kld[fold]={"train":epoch_kld_train,
                        "valid":epoch_kld_test
                            }
        
        torch.save(best_model,"{fname}.pt".format(fname=os.path.join(data_train_dir,"best"+str(fold))))

        tqdm.write("fold {fold:.2f}%".format(
                    fold=fold
                    ))

    np.save(os.path.join(data_train_dir,"loss_results"+'.npy'),fold_loss)
    np.save(os.path.join(data_train_dir,"bce_results"+'.npy'),fold_bce)
    np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),fold_kld)