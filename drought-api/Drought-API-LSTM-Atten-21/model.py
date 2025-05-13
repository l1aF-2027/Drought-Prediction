import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module) : 
    def __init__(self,hidden_dim): 
        super(Attention,self).__init__() 
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self,lstm_output) : 
        attn_weights = self.attn(lstm_output) 
        attn_weights = F.softmax(attn_weights,dim=1) 
        context = torch.sum(attn_weights * lstm_output,dim = 1) 
        return context, attn_weights
class TimeSeriesLSTMAttn(nn.Module) : 
    def __init__(self, time_dim=21, lstm_dim=256, num_layers=2,dropout=0.15,static_dim=30,staticfc_dim=16,hidden_dim = 256,output_size=6) : 
        super(TimeSeriesLSTMAttn,self).__init__() 
        
        self.lstm = nn.LSTM(time_dim,lstm_dim,num_layers = num_layers,batch_first=True,dropout=dropout) 

        self.attention = Attention(lstm_dim) 

        self.static_fc = nn.Sequential(
            nn.Linear(static_dim,staticfc_dim),
            nn.ReLU(),
            nn.Linear(staticfc_dim,staticfc_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim+staticfc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_size)
        )
    def forward(self,x,x_static) : 
        lstm_out,_ = self.lstm(x) 
        context, weights = self.attention(lstm_out)
        static_out = self.static_fc(x_static)
        out = self.fc(torch.cat((context,static_out),1))
        return out 
        