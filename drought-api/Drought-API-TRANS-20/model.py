import torch
from torch import nn

class TimeSeriesTransformer(nn.Module) : 
    def __init__(self,time_dim=21,model_dim=128,seq_len=25*7,n_head=4,dropout = 0.15,num_layers = 2,static_dim=30, staticfc_dim=16,hidden_dim=128,output_size=6) : 
        super(TimeSeriesTransformer,self).__init__() 
        
        self.input_embedding = nn.Linear(time_dim,model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1,seq_len,model_dim))

        # transformer 
        encoder_layer = nn.TransformerEncoderLayer(model_dim,n_head,dropout=dropout,dim_feedforward=model_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers) 
        # static network 
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim,staticfc_dim),
            nn.ReLU(),
            nn.Linear(staticfc_dim,staticfc_dim),
        )
        # final_fc network 
        self.fc = nn.Sequential(
            nn.Linear(model_dim+staticfc_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_size)
        )
    def forward(self,x,x_static) : 
        embedding_input = self.input_embedding(x)
        inputs = embedding_input + self.positional_encoding[:, :x.shape[1], :] 
        
        inputs = inputs.transpose(0, 1)  # (seq_len, batch, model_dim)
        tran_out = self.transformer_encoder(inputs)  # (seq_len, batch, model_dim)
        tran_out = tran_out.transpose(0, 1)  # (batch, seq_len, model_dim)
        tran_out = tran_out.mean(dim=1)
        static_out = self.static_fc(x_static)
        out = self.fc(torch.cat((tran_out,static_out),1))
        return out 