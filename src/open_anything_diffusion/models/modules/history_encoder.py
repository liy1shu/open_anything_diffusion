# Define the history encoder
import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=32,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    ):
        super(HistoryEncoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_layers
        )
        self.input_linear = nn.Linear(input_dim, d_model)
        self.output_linear = nn.Linear(d_model, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, raw_src):
        # raw_src shape: [batch_size, seq_len, input_dim]
        src = raw_src.transpose(0, 1)
        # src shape: [seq_len, batch_size, input_dim]
        src = self.input_linear(src)
        # Transformer expects input of shape: [seq_len, batch_size, d_model]
        transformer_output = self.transformer(src)
        # Pooling over the time dimension to get fixed size output
        pooled_output = torch.mean(transformer_output, dim=0)
        # Final output linear layer to get desired output dimension
        output = self.activation(self.output_linear(pooled_output))
        return output


if __name__ == "__main__":
    # Example usage
    seq_len = 10  # This can vary
    batch_size = 1
    input_dim = 7

    model = HistoryEncoder()
    src = torch.rand(seq_len, batch_size, input_dim)  # Random input
    output = model(src)
    print(output.shape)  # Should be [batch_size, 128]
