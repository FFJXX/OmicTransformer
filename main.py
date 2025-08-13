from train import train
import numpy as np
import torch
folder='ROSMAP'
if __name__ == "__main__": 
    if folder=='BRCA':
        layer=[2,2,2]
        feat_num=[10,10,10]
        embed_dim=[64,64,64]
        num_heads=[16,16,16]
        ep=3
        mask_rate=[0.8,0.8,0.8]
        state=1
        cls=5
        batch_size=102
        epoches=6400
        times=20
    elif folder=='LGG':
        layer=[12,12,12]
        feat_num=[10,10,10]
        embed_dim=[128,128,128]
        num_heads=[8,8,8]
        ep=2
        mask_rate=[0.55,0.55,0.55]
        state=1
        cls=2
        batch_size=72
        epoches=2500
        times=20
    elif folder=='ROSMAP':
        layer=[5,5,5]
        feat_num=[5,5,5]
        embed_dim=[64,64,64]
        num_heads=[16,16,16]
        ep=5
        mask_rate=[0.5,0.5,0.5]
        state=1
        cls=2
        batch_size=49
        epoches=11000
        times=20
    else:
        assert 'error folder'
    if cls>2:
        I='mf1'
    else:
        I='auc'
    acc,wf1,indicator2=train('../'+folder,layer,feat_num,embed_dim, num_heads,ep,mask_rate,state,cls,batch_size,epoches,times)
    print(f'acc:{acc*100:.2f}%,wf1:{wf1*100:.2f}%,{I}:{indicator2*100:.2f}%')