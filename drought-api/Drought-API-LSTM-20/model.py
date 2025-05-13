import torch
from torch import nn

class DroughtNetLSTM(nn.Module):
    def __init__(self, time_dim=20, lstm_dim=256, num_layers=2, dropout=0.15, 
                 static_dim=29, staticfc_dim=16, hidden_dim=256, output_size=6):
        super(DroughtNetLSTM, self).__init__()
        
        # Define LSTM network for time features
        self.lstm = nn.LSTM(
            time_dim,
            lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Define neural network for static features
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, staticfc_dim),
            nn.ReLU(),
            nn.Linear(staticfc_dim, staticfc_dim)
        )
        
        # Define final fully connected layers
        self.final_fc = nn.Sequential(
            nn.Linear(lstm_dim + staticfc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x, x_static):
        """
        Forward pass through the network
        
        Args:
            x: Time series data of shape (batch_size, seq_len, time_dim)
            x_static: Static data of shape (batch_size, static_dim)
            
        Returns:
            out: Output of shape (batch_size, output_size)
        """
        # Process time series data through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take only the last output of the LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Process static data
        static_out = self.static_fc(x_static)
        
        # Concatenate LSTM output and static output
        combined = torch.cat((lstm_out, static_out), 1)
        
        # Final fully connected layers
        out = self.final_fc(combined)
        
        return out