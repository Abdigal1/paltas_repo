from tqdm import tqdm
import numpy as np
import torch
import os
import pickle

def save_dict(dict,path):
    file=open(path,'wb')
    pickle.dump(dict,file)
    file.close()

def load_dict(path):
    file=open(path,'rb')
    dict=pickle.load(file)
    file.close()
    return dict

def security_checkpoint(current_epoch,total_epoch,model,optimizer,loss,PATH):
    torch.save({
        'current_epoch':current_epoch,
        'total_epoch':total_epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss
    },PATH)

def train(model,optimizer,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    rec_d=[]
    con_p_d=[]
    w_p_d=[]
    y_p_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="instances"):
        loss,reconstruction,conditional_prior,w_prior,y_prior=model.ELBO(batch["PhantomRGB"].to(device))
        #loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tqdm.write(
            "total loss {loss:.4f}\treconstruction {rec:.4f}\tconditional_prior {con_p:.4f}\tw_prior {w_p:.4f}\ty_prioir {y_p:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                rec=reconstruction.item(),
                con_p=conditional_prior.item(),
                w_p=w_prior.item(),
                y_p=y_prior.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )


        #SAVE TRAIN DATA
        loss_d.append(loss.item())
        rec_d.append(reconstruction.item())
        con_p_d.append(conditional_prior.item())
        w_p_d.append(w_prior.item())
        y_p_d.append(y_prior.item())
    return loss_d,rec_d,con_p_d,w_p_d,y_p_d


def test(model,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    rec_d=[]
    con_p_d=[]
    w_p_d=[]
    y_p_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="Test"):
        loss,reconstruction,conditional_prior,w_prior,y_prior=model.ELBO(batch["PhantomRGB"].to(device))
        #loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        
        tqdm.write(
            "total loss {loss:.4f}\treconstruction {rec:.4f}\tconditional_prior {con_p:.4f}\tw_prior {w_p:.4f}\ty_prioir {y_p:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                rec=reconstruction.item(),
                con_p=conditional_prior.item(),
                w_p=w_prior.item(),
                y_p=y_prior.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )
        
        #SAVE TEST DATA
        loss_d.append(loss.item())
        rec_d.append(reconstruction.item())
        con_p_d.append(conditional_prior.item())
        w_p_d.append(w_prior.item())
        y_p_d.append(y_prior.item())
    return loss_d,rec_d,con_p_d,w_p_d,y_p_d

def train_test(model,optimizer,train_set,test_set,batch_size,use_cuda,loss_function,epochs,data_train_dir,in_device=None,checkpoint_epoch=0):
    epoch_loss={}
    epoch_rec={}
    epoch_con_p={}
    epoch_w_p={}
    epoch_y_p={}

    epoch_loss_train=[]
    epoch_rec_train=[]
    epoch_con_p_train=[]
    epoch_y_p_train=[]
    epoch_w_p_train=[]

    epoch_loss_test=[]
    epoch_rec_test=[]
    epoch_con_p_test=[]
    epoch_w_p_test=[]
    epoch_y_p_test=[]

    #Is file already exists charge ------------------------------------------------------------------------------------------------------------------------
    if "loss_results.npy" in os.listdir(data_train_dir):
        epoch_rec_train=np.load(os.path.join(data_train_dir,'rec_results.npy'),allow_pickle=True).tolist()['train']
        epoch_rec_test=np.load(os.path.join(data_train_dir,'rec_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_con_p_train=np.load(os.path.join(data_train_dir,'con_p_results.npy'),allow_pickle=True).tolist()['train']
        epoch_con_p_test=np.load(os.path.join(data_train_dir,'con_p_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_y_p_train=np.load(os.path.join(data_train_dir,'y_p_results.npy'),allow_pickle=True).tolist()['train']
        epoch_y_p_test=np.load(os.path.join(data_train_dir,'y_p_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_w_p_train=np.load(os.path.join(data_train_dir,'w_p_results.npy'),allow_pickle=True).tolist()['train']
        epoch_w_p_test=np.load(os.path.join(data_train_dir,'w_p_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_loss_train=np.load(os.path.join(data_train_dir,'loss_results.npy'),allow_pickle=True).tolist()['train']
        epoch_loss_test=np.load(os.path.join(data_train_dir,'loss_results.npy'),allow_pickle=True).tolist()['valid']

    
    best_result=0

    for epoch in tqdm(range(checkpoint_epoch,epochs),desc="Epoch"):

        drop_train=False
        drop_test=False
        if len(train_set)%batch_size==1:
            drop_train=True
        
        if len(test_set)%batch_size==1:
            drop_test=True

        dataloader_train=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=6,drop_last=drop_train)
        dataloader_test=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=6,drop_last=drop_test)

        loss_tr,rec_tr,con_p_tr,w_p_tr,y_p_tr=train(model,optimizer,dataloader_train,use_cuda,loss_function,in_device)
    
        epoch_loss_train.append(np.mean(np.array(loss_tr)))
        epoch_rec_train.append(np.mean(np.array(rec_tr)))
        epoch_con_p_train.append(np.mean(np.array(con_p_tr)))
        epoch_w_p_train.append(np.mean(np.array(w_p_tr)))
        epoch_y_p_train.append(np.mean(np.array(y_p_tr)))
    
        loss_d,rec_d,con_p_d,w_p_d,y_p_d=test(model,dataloader_test,use_cuda,loss_function,in_device)
        #loss_d,bce_d,kld_d=test(model,dataloader_train,use_cuda,loss_function)
        
        if (np.mean(np.array(loss_d)))>best_result:
            best_result=(np.mean(np.array(loss_d)))
            best_model=model.state_dict()
            #TODO: save always
    
        epoch_loss_test.append(np.mean(np.array(loss_d)))
        epoch_rec_test.append(np.mean(np.array(rec_d)))
        epoch_con_p_test.append(np.mean(np.array(con_p_d)))
        epoch_w_p_test.append(np.mean(np.array(w_p_d)))
        epoch_y_p_test.append(np.mean(np.array(y_p_d)))

        tqdm.write("epoch {epoch:.2f}%".format(
                    epoch=epoch
                    ))

        del dataloader_train
        del dataloader_test

        epoch_loss={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }

        epoch_rec={
            "train":epoch_rec_train,
            "valid":epoch_rec_test
        }
        epoch_con_p={
            "train":epoch_con_p_train,
            "valid":epoch_con_p_test
        }
        epoch_w_p={
            "train":epoch_w_p_train,
            "valid":epoch_w_p_test
        }
        epoch_y_p={
            "train":epoch_y_p_train,
            "valid":epoch_y_p_test
        }
        

        np.save(os.path.join(data_train_dir,"loss_results"+'.npy'),epoch_loss)

        np.save(os.path.join(data_train_dir,"bce_results"+'.npy'),epoch_rec)
        np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),epoch_con_p)
        np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),epoch_w_p)
        np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),epoch_y_p)

        #SAVE CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------
        security_checkpoint(current_epoch=epoch,
                            total_epoch=epochs,
                            model=model,
                            optimizer=optimizer,
                            loss=loss_tr,
                            PATH=os.path.join(data_train_dir,"checkpoint.pt")
                            )

    
    return epoch_loss_train,epoch_rec_train,epoch_con_p_train,epoch_w_p_train,epoch_y_p_train,epoch_loss_test,epoch_rec_test,epoch_con_p_test,epoch_w_p_test,epoch_y_p_test,best_model

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

    fold_rec={}
    fold_con_p={}
    fold_w_p={}
    fold_y_p={}


    #Shuffle data
    train_s=int((len(dataset))*0.8)
    test_s=int(len(dataset)-train_s)
    print("train len")
    print(train_s)
    print("test len")
    print(test_s)

    #LOAD INDEXES IS ALREADY EXISTS ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "data_split.pkl" in os.listdir(data_train_dir):
        dict=load_dict(os.path.join(data_train_dir,"data_split.pkl"))
        train_index=dict["train_index"]
        test_index=dict["test_index"]
    else:
        train_index,test_index=torch.utils.data.random_split(range(len(dataset)),[train_s, test_s])
        dataset_split_index={
            "train_index":train_index,
            "test_index":test_index
        }
        save_dict(dataset_split_index,os.path.join(data_train_dir,"data_split.pkl"))

    #train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])
    train_set = torch.utils.data.Subset(dataset, train_index)
    test_set = torch.utils.data.Subset(dataset, test_index)

    drop_train=False
    drop_test=False
    if train_s%batch_size==1:
        drop_train=True
    
    if test_s%batch_size==1:
        drop_test=True

    for fold in tqdm(range(folds),desc="folds"):
        train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])

        ed=model
        #optimizer
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

        #LOAD OPTIMIZER, MODEL, CURRENT EPOCH AND NUMBER OF EPOCHS FROM CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "checkpoint.pt" in os.listdir(data_train_dir):
            checkpoint=torch.load(os.path.join(data_train_dir,"checkpoint.pt"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_epoch=checkpoint["current_epoch"]
            epochs=checkpoint["total_epoch"]
        else:
            checkpoint_epoch=0


        #Epochs
        epoch_loss_train,epoch_rec_train,epoch_con_p_train,epoch_w_p_train,epoch_y_p_train,epoch_loss_test,epoch_rec_test,epoch_con_p_test,epoch_w_p_test,epoch_y_p_test,best_model=train_test(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            test_set=test_set,
            batch_size=batch_size,
            use_cuda=use_cuda,
            loss_function=loss_fn,
            epochs=epochs,
            data_train_dir=data_train_dir,
            in_device=in_device,
            checkpoint_epoch=checkpoint_epoch
        )

        fold_loss[fold]={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }
        
        fold_rec[fold]={
            "train":epoch_rec_train,
            "valid":epoch_rec_test
        }
        fold_con_p[fold]={
            "train":epoch_con_p_train,
            "valid":epoch_con_p_test
        }
        fold_w_p[fold]={
            "train":epoch_w_p_train,
            "valid":epoch_w_p_test
        }
        fold_y_p[fold]={
            "train":epoch_y_p_train,
            "valid":epoch_y_p_test
        }

        torch.save(best_model,"{fname}.pt".format(fname=os.path.join(data_train_dir,"best"+str(fold))))

        tqdm.write("fold {fold:.2f}%".format(
                    fold=fold
                    ))

    np.save(os.path.join(data_train_dir,"fold_loss_results"+'.npy'),fold_loss)
    np.save(os.path.join(data_train_dir,"fold_bce_results"+'.npy'),fold_rec)
    np.save(os.path.join(data_train_dir,"fold_bce_results"+'.npy'),fold_con_p)
    np.save(os.path.join(data_train_dir,"fold_bce_results"+'.npy'),fold_w_p)
    np.save(os.path.join(data_train_dir,"fold_bce_results"+'.npy'),fold_y_p)