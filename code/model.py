import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from .retnet import RetNet
from .GCN import GCN
from sklearn.metrics import r2_score



class RetKcat(nn.Module):
    
    def __init__(self,config=None):
        
        #torch.manual_seed(1220)
        torch.manual_seed(3407)
        super(RetKcat, self).__init__()
        
        if config==None:self.config={'ret_layer': 3, 'out_layer': 3, 'gcn_layer': 2, 'heads': 4, 'hidden_dim': 64, 'ffn_size': 64, 'lr': 0.001, 'WD': 0.0005}
        else:self.config=config

        for i in self.config:
            self.__dict__[i]=self.config[i]

        self.seq_ebd1=nn.Embedding(9000,int(self.hidden_dim/2))
        self.seq_ebd2=nn.Embedding(9000,int(self.hidden_dim/2))
        self.retnet=RetNet(layers=self.ret_layer,hidden_dim=self.hidden_dim,ffn_size=self.ffn_size,heads=self.heads)

        self.mol_ebd=nn.Embedding(4096,self.hidden_dim)
        self.act=nn.LeakyReLU()
        self.GCN = nn.ModuleList([GCN(self.hidden_dim,self.hidden_dim) for _ in range(self.gcn_layer)])
        
        self.out_Lrelu=nn.LeakyReLU()
        self.out=nn.ModuleList([nn.Linear(self.hidden_dim,self.hidden_dim) for _ in range(self.out_layer)])

        self.trans=nn.Linear(self.hidden_dim,1)

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.optimizer= Adam(self.parameters(),lr=self.lr, weight_decay=self.WD)
        
    def gcn(self,x,A):

        x=self.mol_ebd(x)
        for i in range(self.gcn_layer):
            x=self.act(self.GCN[i](x,A))

        return sum(x)

    def ret(self,seq):
        
        x=torch.cat((self.seq_ebd1(seq[0]),self.seq_ebd2(seq[1])),dim=1)
        x=torch.unsqueeze(x,0)
        x=self.retnet(x)
        x=torch.squeeze(x,0)
        
        return sum(x)
        
    def forward(self,inputs):

        seq,mol,A=inputs

        Xm=torch.unsqueeze(self.gcn(mol,A),0)
        Xs=torch.unsqueeze(self.ret(seq),0)
   
        X=torch.cat((Xm,Xs))

        for i in range(self.out_layer):
            X=self.out_Lrelu(self.out[i](X))
        
        X=self.trans(X)
        
        return torch.mean(X).unsqueeze(0)

    def Train(self, dataset):
        
        '''return float(loss_total),float(rmse),r2'''
        
        self.train()
        loss_total = 0
        predicted_values_list=[]
        correct_values_list=[]
        
        for data in dataset:
            inp,correct_value=self.dataloader(data)
            
            predicted_value = self.forward(inp)

            loss = F.mse_loss(predicted_value,correct_value)
            correct_values_list.append(correct_value.cpu().detach().numpy())
            predicted_values_list.append(predicted_value.cpu().detach().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total+=loss.cpu().detach().numpy()
            del loss
        
        correct_values_list=np.array(correct_values_list)
        predicted_values_list=np.array(predicted_values_list)
        rmse=torch.sqrt(F.mse_loss(torch.tensor(correct_values_list),torch.tensor(predicted_values_list)))
        r2=r2_score(correct_values_list,predicted_values_list)
        
        return float(loss_total),float(rmse),r2
    
    def Test(self, dataset):
        
        '''return float(MAE),float(rmse),r2'''
        
        self.eval()
        np.random.shuffle(dataset)
        
        SAE = 0
        predicted_values_list=[]
        correct_values_list=[]
        
        for data in dataset :
            inp,correct_value=self.dataloader(data)
            
            predicted_value = self.forward(inp)
            
            SAE += sum(torch.abs(predicted_value-correct_value))
            correct_values_list.append(correct_value.cpu().detach().numpy())
            predicted_values_list.append(predicted_value.cpu().detach().numpy())

        correct_values_list=np.array(correct_values_list)
        predicted_values_list=np.array(predicted_values_list)
        rmse=torch.sqrt(F.mse_loss(torch.tensor(correct_values_list),torch.tensor(predicted_values_list)))
        r2=r2_score(correct_values_list,predicted_values_list)
        MAE = SAE / len(dataset)
        
        return float(MAE),float(rmse),r2

    def dataloader(self,data):
        inp=[[torch.tensor(i,dtype=torch.int).to(self.device) for i in data[0]]]
        inp.append(torch.tensor(data[1],dtype=torch.int).to(self.device))
        inp.append(torch.tensor(data[2],dtype=torch.float).to(self.device))
        return inp,torch.tensor(data[3],dtype=torch.float).to(self.device).unsqueeze(0)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
  