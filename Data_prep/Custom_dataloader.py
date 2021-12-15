import os
import torch
import numpy as np
import glob
from skimage import io
import matplotlib.pyplot as plt
import PIL.Image

class Dataset_direct(torch.utils.data.Dataset):
    def __init__(self,root_dir,ImType=['PhantomRGB', 'SenteraRGB', 'SenteraNIR'],
                 days='*',
                 months='*',
                 Trees_col='*',
                 Trees_fil='*',
                 Intersec=False,
                 transform=None,
                 retrieve_img=True):
            """Data loader
            inputs:
            -root_dir(str): Directory that contains all the directories per tree
            -ImType(list of str): type of images to be used in posterior processing, ex: ['PhantomRGB', 'SenteraRGB', 'SenteraNIR']
            -days(list of integers): specific dates or '*' if date selection is no needed, ex: [13,19]
            -months(list of str):specific months or '*' if month selection is not needed, ex: ['junio','mayo']
            -Trees_col=(list of integers): specific numbers or '*' if date selection is no needed, ex: [13,19]
            -Trees_fil(list of str):specific rows or '*' if month selection is not needed, ex: ['A','C']
            -Intersec(bool):True if regular dataset in the selecter data types,
            -transform=None
            Outputs:

            Troubleshooting:
            -if number of NIR images and RGB images from sentera do not match
            """
            'Initialization'

            self.toID=np.vectorize(lambda d:(("_").join(np.array((os.path.split(d)[1]).split("_"))[np.array([0,1,2,-3,-1])])).split(".")[0])
            
            self.ImType=ImType
            self.retrieve_img=retrieve_img
            
            Map_cols={'A':'N',
                    'B':'P',
                    'C':'K',
                    'D':'Control',
                    'E':'H50%',
                    'F':'H75%',
                    'G':'Control'}

            Map_fils=np.hstack((
                (np.arange(1,22)).reshape(3,-1).astype('object'),
                np.array([['Exceso'],['Control'],['Deficiencia']]).astype('object')
            ))
            
            self.DATA={
                'PhantomRGB':{
                    'data':np.array([]).astype('<U21'),
                    'ID':np.array([]).astype('<U21'),
                    'Place_label':np.array([]).astype('<U21'),
                    'Date':np.array([]).astype('<U21')
                },
                'SenteraRGB':{
                    'data':np.array([]).astype('<U21'),
                    'ID':np.array([]).astype('<U21'),
                    'Place_label':np.array([]).astype('<U21'),
                    'Date':np.array([]).astype('<U21')
                },
                'SenteraNIR':{
                    'data':np.array([]).astype('<U21'),
                    'ID':np.array([]).astype('<U21'),
                    'Place_label':np.array([]).astype('<U21'),
                    'Date':np.array([]).astype('<U21')
                },
                'SenteraMASK':{
                    'data':np.array([]).astype('<U21'),
                    'ID':np.array([]).astype('<U21'),
                    'Place_label':np.array([]).astype('<U21'),
                    'Date':np.array([]).astype('<U21')
                }
            }
            
            for Type in ImType:
                data,ID=self.process_dirs(
                    root_dir=root_dir,
                    Type=Type,
                    days=days,
                    months=months,
                    Trees_col=Trees_col,
                    Trees_fil=Trees_fil
                )
                self.DATA[Type]['data']=data
                self.DATA[Type]['ID']=ID
            
            self.IDs=np.vectorize(lambda DATA,T:DATA[T]['ID'],otypes=[object],signature="(),()->()")(self.DATA,
                                                                                               np.array(ImType))
            
            uIDs=np.vectorize(lambda Id:np.unique(Id),otypes=[object])(self.IDs)
            self.aID=uIDs[0]
            for i in range(len(uIDs)-1):
                self.aID=np.union1d(uIDs[0],uIDs[1])
            #Only intersections
            if Intersec:
                
                uIDs=np.vectorize(lambda ID:np.unique(ID),otypes=[object])(self.IDs)
                inters=uIDs[0]
                for i in range(len(uIDs)-1):
                    inters=np.intersect1d(inters,uIDs[i+1])
            
                if inters.shape==(0,):
                    print("no hay elementos en común")
                else:
                    for Type in ImType:
                        whr=np.vectorize(lambda ID,selec: ID in selec,signature="(),(j)->()")(self.DATA[Type]['ID'],inters)
                        self.DATA[Type]['ID']=self.DATA[Type]['ID'][whr]
                        self.DATA[Type]['data']=self.DATA[Type]['data'][whr]
            
            #PROCESS LABELS
            
            for Type in ImType:
                #print(Type)
                #print(self.DATA[Type]['ID'][0])
                self.DATA[Type]['Place_label']=np.vectorize(self.get_labels)(self.DATA[Type]['ID'])
                self.DATA[Type]['Date']=np.vectorize(lambda iD: ("_").join((iD).split("_")[0:3]))(self.DATA[Type]['ID'])
                
            self.landmarks_frame_PRGB = self.DATA['PhantomRGB']['data']
            self.landmarks_frame_SRGB = self.DATA['SenteraRGB']['data']
            self.landmarks_frame_SNIR = self.DATA['SenteraNIR']['data']
            self.landmarks_frame_SMASK = self.DATA['SenteraMASK']['data']
                
            self.transform = transform

    def __len__(self):
        #return len(self.landmarks_frame_PRGB),len(self.landmarks_frame_SRGB),len(self.landmarks_frame_SNIR)
        return len(self.aID)

    def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            idxID=self.aID[idx]
            images_dir=np.vectorize(lambda data,TT,idx:data[TT]["data"][data[TT]["ID"]==idx],
                                    otypes=[object],
                                    signature="(),(),()->()")(self.DATA,np.array(self.ImType),idxID)
            
            landmarks=np.vectorize(lambda data,TT,idx:data[TT]["Place_label"][data[TT]["ID"]==idx],
                                    otypes=[object],
                                    signature="(),(),()->()")(self.DATA,np.array(self.ImType),idxID)
            
            if self.retrieve_img:
                images=np.vectorize(self.special_imread,otypes=[object])(images_dir)

            sample={}
            for i in range(len(self.ImType)):
                #print("image")
                #print(images_dir[i][0])
                if self.retrieve_img:
                    sample[self.ImType[i]]=images[i]
                    if (images[i].shape!=(1,1))and(images_dir[i][0].endswith(".jpg")):
                        sample[self.ImType[i]+"_metadata"]=self.get_metadata(images_dir[i][0])
                    else:
                        sample[self.ImType[i]+"_metadata"]=None
                
            sample["Date"]=("_").join(idxID.split("_")[:3])
            sample["Place"]=("_").join(idxID.split("_")[3:])
            sample["landmarks"]=landmarks[0][0]
            if self.transform:
                  sample=self.transform(sample)

            return sample
            

    def special_imread(self,im):
        if im.shape!=(0,):
            oim=io.imread(im[0])
        else:
            oim=np.array([[0]])
        return oim

    def process_dirs(self,root_dir,months,days,Trees_col,Trees_fil,Type):
        data=np.vectorize(lambda days,mth,Trees_col:glob.glob(os.path.join(root_dir,"*/"+
                                                            Type+"/"+
                                                            str(days)+"_"+
                                                            mth+"_*__arbol_"+
                                                            str(Trees_col)+"_fila_"+
                                                            str(Trees_fil)+
                                                            ".*")),
                                   signature="(),(),()->()",
                                   otypes=[object])(
            np.array(days),
            np.array(months),
            np.array(Trees_col)
        )
        
        if days!='*' or months!='*' or Trees_col!='*':
            data=np.concatenate(data)
        else:
            data=np.array(data.reshape(1,)[0])
        if data.shape!=(0,):
            ID=self.toID(data)
        else:
            ID=np.array([])
        return data,ID
        
    
    #def intersected():
    
    def get_labels(self,IDs):
        Map_cols={'A':'N','B':'P','C':'K',
                  'D':'Control','E':'H50%','F':'H75%','G':'Control'}
        Map_fils=np.hstack(((np.arange(1,22)).reshape(3,-1).astype('object'),
                        np.array([['Exceso'],['Control'],['Deficiencia']]).astype('object')))
        Type=Map_cols[IDs.split('.')[0].split("_")[-1]]
        Num=int(IDs.split('.')[0].split("_")[-2])
        if Type in ['N','P','K']:
            Suf=Map_fils[np.vectorize(lambda ran,n:n in ran,signature="(j),()->()")(Map_fils[:,:-1],Num),-1]
            Type=(Type+"_"+Suf)[0]
        return Type

    def get_metadata(self,direc):
        img=PIL.Image.open(direc)
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        exif['GPSInfo'] = {
            PIL.ExifTags.GPSTAGS[k]: v
            for k, v in exif['GPSInfo'].items()
            if k in PIL.ExifTags.GPSTAGS 
        }
        return exif