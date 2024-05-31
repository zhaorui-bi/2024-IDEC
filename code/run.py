import sys
import torch
import time
import pandas as pd
from .model import *

def dataloader(self,data):
    inp=[[torch.tensor(i,dtype=torch.int).to(self.device) for i in data[0]]]
    inp.append(torch.tensor(data[1],dtype=torch.int).to(self.device))
    inp.append(torch.tensor(data[2],dtype=torch.float).to(self.device))
    return inp,data[3]

def predict(data:list,modestate='codes/tr30.modelstate'):
    config={'ret_layer': 3, 'out_layer': 3, 'gcn_layer': 2, 'heads': 4, 'hidden_dim': 64, 'ffn_size': 64, 'lr': 0.001, 'WD': 0.0005}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rk=RetKcat(config).to(device)
    
    rk.load_state_dict(torch.load(modestate))

    
    time0=time.time()
    datalist=[]
    N=len(data)
    tag0=0
    n=0
    
    for i in data:
        n+=1

        if len(i)==5:
            name=i[0]
            i=i[1:]
        else:name=str(n)
        
        tag1=round(n/N*100)
        if tag1>tag0:
            print(tag1,'%')
            tag0=tag1
        inp,interaction=rk.dataloader(i)
        pred=rk.forward(inp).cpu()

        datalist.append([name,pred.item(),interaction.item()])

    print('forwaded',N,'cost',round(time.time()-time0,2),'s')
    pd.DataFrame(datalist,columns=['id','prediction','correct']).to_csv('output.csv',index=False)