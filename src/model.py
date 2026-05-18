import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Deep Q Network Model - extraction network followed by a Q-network

class QNetwork(nn.Module):
    def __init__(self, hidden_lstm_size, hidden_fc_size, output_size):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_lstm_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_lstm_size+3, hidden_fc_size, bias=True)
        self.fc2 = nn.Linear(hidden_fc_size, output_size, bias=True)
    
    def forward(self, x):
        dim_1 = x.dim() == 1
        if dim_1:  # Check wether the tensor is 1D
            x = x.unsqueeze(0)

        prices_derivative = x[:, 3:].unsqueeze(-1)  # Shape: (batch_size, PRICES_HISTORY_LEN-1, 1)
        static_features = x[:, :3]  # Shape: (batch_size, 3)

        _, (h_n, _) = self.lstm(prices_derivative)  # h_n shape: (1, batch_size, hidden_lstm_size)
        lstm_out = h_n[-1]  # Shape: (batch_size, hidden_lstm_size)

        x = torch.cat([lstm_out, static_features], dim=1)  # Shape: (batch_size, hidden_lstm_size+3)
        x = F.relu(self.fc1(x)) # Shape: (batch_size, hidden_fc_size)
        x = self.fc2(x)     # Shape: (batch_size, output_size)

        if dim_1:
            x = x.squeeze(0)

        return x
