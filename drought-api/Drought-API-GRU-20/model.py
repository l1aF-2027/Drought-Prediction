import torch
from torch import nn

class DroughtNetGRU(nn.Module) : 
    def __init__(self, time_dim=21, gru_dim=256, num_layers=2,dropout=0.15,static_dim=30,staticfc_dim=16,hidden_dim = 256,output_size=6) : 
        super(DroughtNetGRU,self).__init__() 
        # define gru_net for time_features
        self.gru = nn.GRU(time_dim,gru_dim,num_layers=num_layers,batch_first=True,dropout = dropout) 
        # define nn_net for static_features
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim,staticfc_dim),
            nn.ReLU(),
            nn.Linear(staticfc_dim,staticfc_dim)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(gru_dim+staticfc_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_size)
        )
    def forward(self,x,x_static) : 
        gru_out,_ = self.gru(x) 
        gru_out = gru_out[:, -1, :]
        static_out = self.static_fc(x_static)
        out = self.final_fc(torch.cat((gru_out, static_out), 1))
        return out 