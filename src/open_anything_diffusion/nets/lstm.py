import torch
import torch.nn as nn

class LSTMAggregator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        outputs, (h_n, c_n) = self.lstm(x)
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        # c_n: (num_layers, batch_size, hidden_size)
        
        # You can use the last hidden state h_n[-1] as the aggregated representation
        # h_n[-1]: (batch_size, hidden_size)
        # return h_n
        return outputs