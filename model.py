import torch
import torch.nn as nn
device='cuda:0'

class EncoderLayer(nn.Module):
    def __init__(self,dff,dmodel,head,rate=0.1):
        super().__init__()
        self.dmodel=dmodel
        self.head=head
        self.multiattention=nn.MultiheadAttention(dmodel,head,dropout=rate)
        self.L1=nn.Linear(dmodel,dff)
        self.act=nn.ELU()
        self.D1=nn.Dropout(rate)
        self.L2=nn.Linear(dff,dmodel)
        self.D2=nn.Dropout(rate)
        self.D=nn.Dropout(rate)
        self.ln1=nn.LayerNorm(dmodel,eps=1e-5)
        self.ln2=nn.LayerNorm(dmodel,eps=1e-5)
        
    def forward(self,x):
        mo,weight=self.multiattention(x,x,x,need_weights=True)
        mo=self.D(mo)
        mo=self.ln1(x+mo)
        fo=self.L2(self.D1(self.act(self.L1(mo))))
        fo=self.D2(fo)
        fo=self.ln2(fo+mo)
        return fo,weight

class OmicTransformer(nn.Module):
    def __init__(self, dims, embed_dim, num_heads, num_layers, num_classes,n_feat,ep):
        super().__init__()
        self.TS=nn.ModuleList()
        self.Res=nn.ModuleList()
        self.emb=nn.ParameterList()
        self.n_feat=n_feat
        self.transformer_encoder=nn.ModuleList()
        self.Confidence=nn.ModuleList() 
        self.TCP=nn.ModuleList() 
        for n,dim in enumerate(dims):
            TS=nn.ModuleList()
            Res=nn.ModuleList()
            emb=nn.Parameter(torch.empty((1,1,embed_dim[n])),requires_grad=True) 
            nn.init.xavier_normal_(emb.data)
            for i in range(n_feat[n]):
                p=nn.Linear(dim,embed_dim[n])
                TS.append(nn.Sequential(p,nn.ELU()))
                Res.append(nn.Sequential(nn.Linear(embed_dim[n],num_classes)))
            self.TS.append(TS)
            self.Res.append(Res)
            self.emb.append(emb)
            self.transformer_encoder.append(nn.ModuleList([EncoderLayer(int(embed_dim[n]*ep),embed_dim[n], num_heads[0])  for i in range(num_layers[n])]))
            self.Confidence.append(nn.Sequential(nn.Linear(embed_dim[n],1),nn.Sigmoid()))
            self.TCP.append(nn.Sequential(nn.Linear(embed_dim[n],num_classes)))

        self.layer=num_layers
        self.fc_fin = nn.Linear(embed_dim[0], num_classes)
    def R_loss(self,pred,label):
        loss=0.
        for view in range(len(pred)):
            for f in range(pred[view].shape[0]):
                p=pred[view][f]
                loss+=nn.functional.cross_entropy(p,label)
        return loss
    def Masked_loss(self,feat,W):
        loss=0.
        for view in range(len(feat)):
            anchor=feat[view]
            sam=W[view]
            loss+=torch.sum(torch.exp(torch.ones(anchor.shape[:2])).to(device)-torch.cosine_similarity(anchor,sam,dim=2).exp())
        return loss/(anchor.shape[1])
    def conf_loss(self,cls,conf,label_onehot):
        label=torch.argmax(label_onehot,dim=1)
        loss=0.
        cls_loss=0.
        for i in range(len(cls)):
            pred=nn.functional.softmax(cls[i],dim=1)
            tar=torch.gather(input=pred,dim=1,index=label.unsqueeze(dim=1)).view(-1)
            C=conf[i].view(-1)
            loss+=nn.functional.mse_loss(C,tar)
            cls_loss+=nn.functional.cross_entropy(cls[i],label)
        loss+=cls_loss
        return loss
        
    def forward(self,X,state,rate,training=True):
        RES=[]
        feat_res=[]
        INTE=[]
        WEIGHT=[]
        f_D=[]
        CONF,CLS=[],[]
        fin=0
        for n,x in enumerate(X):
            for i in range(self.n_feat[n]):
                t=self.TS[n][i](x)             
                if i==0:  
                    nx=t.unsqueeze(0)
                    res=self.Res[n][i](t).unsqueeze(0)
                else:
                    nx=torch.concat((nx,t.unsqueeze(0)),dim=0)
                    res=torch.concat((res,self.Res[n][i](t).unsqueeze(0)),dim=0)
                    
                p=nn.functional.dropout(x,rate[n])
                w=self.TS[n][i](p) 
                if i==0:            
                    W=w.unsqueeze(0)
                else:
                    W=torch.concat((W,w.unsqueeze(0)),dim=0)
                    
            feat_res.append(nx)
            if training:
                f_D.append(W)
                RES.append(res)

            nx=torch.concat((nx,self.emb[n].repeat(1,nx.shape[1],1)),dim=0)
            for i in range(self.layer[n]):
                nx,weight=self.transformer_encoder[n][i](nx)
            INTE.append(nx[-1])
            WEIGHT.append(weight)

            if state==1:
                conf=self.Confidence[n](INTE[n])
                cls=self.TCP[n](INTE[n])
                fin+=INTE[n]*conf
                CONF.append(conf)
                CLS.append(cls)
            else:
                fin+=INTE[n]

        y = self.fc_fin(fin) 

            
        return y,feat_res,f_D,WEIGHT,RES,CLS,CONF,INTE