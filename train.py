import torch
import pandas as pd
import pickle
import torch.nn as nn
device='cuda:0'
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import matplotlib.pyplot as plt
import sys
import random
from model import EncoderLayer,OmicTransformer

from torch.utils.data import Sampler
class StratifiedSampler(Sampler):
    def __init__(self, data_len, labels,batch_size,cls_num, seed=None):
        self.data_len = data_len
        self.labels = labels
        self.seed = seed
        self.batch_size=batch_size
        self.cls_num=cls_num            
        if self.seed is not None:
            np.random.seed(self.seed)
        self.proportion,self.label_num=self._proportion()
        self.indice=self._stratify()
    def _proportion(self):
        label_to_indices={}
        for idx, label in enumerate(self.labels):
            p=torch.argmax(label).item()
            if p not in label_to_indices:
                label_to_indices[p] = 0
            label_to_indices[p]+=1
        res=list(np.ndarray([self.cls_num],dtype=np.int32))
        for label,num in label_to_indices.items():
            res[label]=num
        res_num=torch.tensor(res,dtype=torch.float32)
        res_p=res_num/len(self.labels)
        return res_p,res_num
    def _stratify(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            p=torch.argmax(label).item()
            if p not in label_to_indices:
                label_to_indices[p] = np.ndarray([0],dtype=np.int32)
            label_to_indices[p]=np.append(label_to_indices[p],idx)
        indices = []

        for label, idxs in label_to_indices.items():
            np.random.shuffle(idxs)
        for i in range(self.data_len//self.batch_size):
            for label, idxs in label_to_indices.items():
                p=int(torch.round(self.batch_size*self.proportion[label]))     
                temp=idxs[p*i:p*(i+1)]
                indices.extend(temp)
        for label, idxs in label_to_indices.items():
            p=int(torch.round(self.batch_size*self.proportion[label]))
            temp=idxs[p*(self.data_len//self.batch_size):]
            indices.extend(temp)
        return indices

    def __iter__(self):
        return iter(self.indice)

    def __len__(self):
        return self.data_len

def similarity(X):#calculate the cosine similarity between samples
    shape=X.shape
    norm=torch.norm(X,dim=1)
    norm=torch.where(norm==0,1,norm)
    norm=torch.reshape(norm,(shape[0],1))
    X=X/norm
    p=(X@X.T)
    A=p.sum()
    return (A-shape[0])/(shape[0]**2-shape[0])

def preprocess(data,cls,batch_size):
    mrna_train=pd.read_csv(f'{data}/1_tr.csv',sep=',',header=None)
    mrna_test=pd.read_csv(f'{data}/1_te.csv',sep=',',header=None)
    dna_train=pd.read_csv(f'{data}/2_tr.csv',sep=',',header=None)
    dna_test=pd.read_csv(f'{data}/2_te.csv',sep=',',header=None)
    mirna_train=pd.read_csv(f'{data}/3_tr.csv',sep=',',header=None)
    mirna_test=pd.read_csv(f'{data}/3_te.csv',sep=',',header=None)
    train_label=pd.read_csv(f'{data}/labels_tr.csv',sep=',',header=None)
    test_label=pd.read_csv(f'{data}/labels_te.csv',sep=',',header=None)
    dna_dim=dna_train.shape[1]
    mrna_dim=mrna_train.shape[1]
    mirna_dim=mirna_train.shape[1]
    dna_train=torch.from_numpy(dna_train.to_numpy()).to(torch.float32)
    mrna_train=torch.from_numpy(mrna_train.to_numpy()).to(torch.float32)
    mirna_train=torch.from_numpy(mirna_train.to_numpy()).to(torch.float32)
    train_label=torch.from_numpy(train_label.to_numpy()).to(torch.float32)

    dna_test=torch.from_numpy(dna_test.to_numpy()).to(torch.float32)
    mrna_test=torch.from_numpy(mrna_test.to_numpy()).to(torch.float32)
    mirna_test=torch.from_numpy(mirna_test.to_numpy()).to(torch.float32)
    test_label=torch.from_numpy(test_label.to_numpy()).to(torch.float32)

    sample_size_train=mrna_train.shape[0]
    sample_size_test=mrna_test.shape[0]

    train_label=torch.reshape(train_label,(-1,)).to(int)
    train_label=nn.functional.one_hot(train_label,num_classes=cls).to(torch.float32)

    test_label=torch.reshape(test_label,(-1,)).to(int)
    test_label=nn.functional.one_hot(test_label,num_classes=cls).to(torch.float32)

    train_sampler = StratifiedSampler(dna_train.shape[0],train_label,batch_size,cls,seed=42)

    train_dataset=[((mrna_train[i],dna_train[i],mirna_train[i]),train_label[i]) for i in range(sample_size_train)]
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,drop_last=False)
    test_dataset=[((mrna_test[i],dna_test[i],mirna_test[i]),test_label[i]) for i in range(sample_size_test)]
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=sample_size_test,shuffle=False)
    return train_dataloader,test_dataloader,(mrna_dim,dna_dim,mirna_dim),((mrna_train,dna_train,mirna_train),(mrna_test,dna_test,mirna_test))

@torch.no_grad()
def Test(model,test_dataloader,state,cls_num,multi,missing=False):
    model.eval()
    total=0
    p=0
    wf1=0
    indicator2=0
    for data in test_dataloader:
        x,label=data[0],data[1]
        label=label.to(device)
        x1,x2,x3=x[0],x[1],x[2]
        if missing:
            for b in range(x1.shape[0]):
                num=int(torch.randint(0,3,(1,)))#0-2
                if num==0:
                    continue
                rand_index=torch.randint(0,3,(num,))#random list
                for i in rand_index:
                    if i==0:
                        x1[b]=torch.zeros((x1[b].shape[0]))
                    elif i==1:
                        x2[b]=torch.zeros((x2[b].shape[0]))
                    elif i==2:
                        x3[b]=torch.zeros((x3[b].shape[0]))
        x1,x2,x3=x1.to(device),x2.to(device),x3.to(device)
        y,_,_,weight,_,_,conf,X=model([x1,x2,x3],state,[0,0,0],training=False)

        total+=(torch.argmax(y,dim=1)==torch.argmax(label,dim=1)).sum()
        label=torch.argmax(label,dim=1).cpu()
        y=y.cpu()
        if multi==False:
            indicator2+=roc_auc_score(label,y[:,1])*label.shape[0]
            y=torch.argmax(y,dim=1)
        else:
            y=torch.argmax(y,dim=1)
            indicator2+=f1_score(label,y,average='macro')*label.shape[0]
        p+=label.shape[0]
        wf1+=f1_score(label,y,average='weighted')*label.shape[0]
        
    acc_cls=[]
    for i in range(cls_num):
        t=(label==i).sum()
        a=(y[label==i]==i).sum()
        acc_cls.append(a/t*100)
        
    return total/p,wf1/p,indicator2/p,acc_cls

def train(data,layer,feat_num,embed_dim, num_heads,ep,mask_rate,state,cls,batch_size,epoches,times):
    if cls>2:
        I='mf1'
    else:
        I='auc'
    train_dataloader,test_dataloader,dims,_=preprocess(data,cls,batch_size)
    ACC,WF1,I2=0,0,0
    print('Training')
    for time in range(1,times+1):
        print(f'Time {time}')
        omic_t=OmicTransformer(dims,embed_dim,num_heads,layer,cls,feat_num,ep)
        omic_t=omic_t.to(device=device)
        Adam=torch.optim.Adam(omic_t.parameters(),lr=3e-6)#3e-6
        L=99999.
        for epoch in range(1,epoches+1):
            omic_t.train()
            l=0
            acc=0
            S=0
            for batch,tdata in enumerate(train_dataloader):
                Adam.zero_grad()
                x,label=tdata[0],tdata[1]
                x1,x2,x3=x[0],x[1],x[2]
                #modality dropout
                for b in range(x1.shape[0]):
                    num=int(torch.randint(0,3,(1,)))#0-2
                    if num==0:
                        continue
                    rand_index=torch.randperm(3)[:num]#random list
                    for i in rand_index:
                        if i==0:
                            x1[b]=torch.zeros((x1[b].shape[0]))
                        elif i==1:
                            x2[b]=torch.zeros((x2[b].shape[0]))
                        elif i==2:
                            x3[b]=torch.zeros((x3[b].shape[0]))
                #end
                label=label.to(device)
                x1=x1.to(device)
                x2=x2.to(device)
                x3=x3.to(device)
                y,feat,W,weight,Res,C,conf,omic_feat=omic_t([x1,x2,x3],state,mask_rate,training=True)
                tcploss=omic_t.conf_loss(C,conf,label)
                mloss=omic_t.Masked_loss(feat,W)
                rloss=omic_t.R_loss(Res,label)
                loss=nn.functional.cross_entropy(y,label,reduction='mean')+mloss+tcploss+rloss
                loss.backward()
                Adam.step()     
                l+=loss.item()
                acc+=(torch.argmax(y,dim=1)==torch.argmax(label,dim=1)).sum()
                S+=label.shape[0]
            l/=(batch+1)  
            if l<L:
                L=l
                torch.save(omic_t,f'{data}_t.pt')
            if epoch%100==0:
                p=torch.load(f'{data}_t.pt',weights_only=False)
                acc,wf1,indicator2,acc_class=Test(p,test_dataloader,state,cls,cls>2,missing=False)
                print(f'epoch:{epoch} loss:{l:.2f},acc:{acc*100:.2f}%,wf1:{wf1*100:.2f}%,{I}:{indicator2*100:.2f}%')
        p=torch.load(f'{data}_t.pt',weights_only=False)
        torch.save(p,f'{data}_{time}.pt')
        ACC+=acc
        WF1+=wf1
        I2+=indicator2
    return ACC/times,WF1/times,I2/times