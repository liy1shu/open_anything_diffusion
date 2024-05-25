# Define the history encoder
import math

import lightning as L
import torch
import torch.nn as nn
from torch.nn import init
import torch_geometric.data as tgd
from open_anything_diffusion.nets.lstm import LSTMAggregator


# Yishu's old old old version - history : grasp point & direction & outcome
# class PDOHistoryEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim=7,
#         output_dim=32,
#         d_model=128,
#         nhead=4,
#         num_layers=2,
#         dim_feedforward=256,
#         batch_norm=False,
#     ):
#         super(PDOHistoryEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.d_model = d_model
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_layers
#         )
#         # self.lstm_aggregator = LSTMAggregator(input_size=input_dim, 
#         #                                       hidden_size=d_model)                                      
      
#         self.input_linear = nn.Linear(input_dim, d_model)
#         self.output_linear = nn.Linear(d_model, output_dim)
#         self.activation = nn.Sigmoid()

#     def forward(self, raw_src):
#         # raw_src shape: [batch_size, seq_len, input_dim]
#         src = raw_src.transpose(0, 1)
#         # src shape: [seq_len, batch_size, input_dim]
#         src = self.input_linear(src)
#         # Transformer expects input of shape: [seq_len, batch_size, d_model]
#         lstm_output = self.lstm_aggregator(src)
#         # Pooling over the time dimension to get fixed size output
#         pooled_output = torch.mean(lstm_output, dim=0)
#         # Final output linear layer to get desired output dimension
#         output = self.activation(self.output_linear(pooled_output))
#         return output


def get_history_batch(batch, device):
    """Extracts a single batch of the history data for encoding,
    because each history element is processed separately."""
    has_history_ids = []
    no_history_ids = []
    history_datas = []
    for id, data in enumerate(batch.to_data_list()):
        # if data.K == 0:  # No history
        #     no_history_ids.append(id)
        #     continue
        # has_history_ids.append(id)
        history_data = []
        # Get start/end positions based on lengths.

        # HACK: remove once the data has been regenerated...
        if len(data.history.shape) == 3:
            data.history = data.history.reshape(-1, 3)
            data.flow_history = data.flow_history.reshape(-1, 3)

        N = data.pos.shape[0]  # num of points
        if hasattr(data, "lengths"):
            ixs = [0] + data.lengths.cumsum(0).tolist()
        else:
            ixs = [(i * N) for i in range(data.K + 1)]
        for i in range(len(ixs) - 1):
            history_data.append(
                tgd.Data(
                    x=data.flow_history[ixs[i] : ixs[i + 1]],
                    pos=data.history[ixs[i] : ixs[i + 1]],
                )
            )

        history_datas.extend(history_data)
    # if len(history_datas) == 0:
    #     return no_history_ids, has_history_ids, None  # No has_history batch
    # return no_history_ids, has_history_ids, tgd.Batch.from_data_list(history_datas)

    return tgd.Batch.from_data_list(history_datas).to(device) if len(history_datas) > 0 else []


def history_latents_to_nested_list(batch, history_latents):
    """Converting history latents from stacked form to nested list"""
    # breakpoint()
    datas = batch.to_data_list()
    history_lengths = [0] + [data.K.item() for data in datas]
    # history_lengths = [0] + [1 for data in datas]
    ixs = torch.tensor(history_lengths).cumsum(0).tolist()

    post_encoder_latents = []
    for i, data in enumerate(datas):
        post_encoder_latents.append(history_latents[ixs[i] : ixs[i + 1]])

    return post_encoder_latents


# Previous flow history
class HistoryEncoder(L.LightningModule):
    def __init__(
        self,
        history_dim=128,
        history_len=1,
        batch_norm=False,
        transformer=True,
        repeat_dim=True,
    ):
        super(HistoryEncoder, self).__init__()
        self.point_cnts = 1200
        self.history_len = history_len
        assert self.history_len == 1, "currently only supports 1 previous step history"
        self.history_dim = history_dim

        if batch_norm:
            from rpad.pyg.nets import pointnet2 as pnp
        else:
            import open_anything_diffusion.models.modules.pn2 as pnp
        self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=history_dim)
        self.no_history_embedding = nn.Parameter(
            torch.randn(history_dim), requires_grad=True
        )
        self.repeat_dim = repeat_dim
        self.transformer = transformer

        # Start token for no history
        self.start_token = nn.Parameter(torch.empty((history_dim,)).float(), requires_grad=True)
        bound = 1/math.sqrt(history_dim)
        init.uniform_(self.start_token, -bound, bound)

        if self.transformer:
            self.transformer = LSTMAggregator(input_size=history_dim, 
                                              hidden_size=history_dim)                                      
      
            #nn.Transformer(d_model=history_dim)

    def forward(self, batch):

        history_batch = get_history_batch(batch, self.device)

        if len(history_batch) > 0:
            history_embeds = self.prev_flow_encoder(history_batch)
        else:
            history_embeds = torch.empty((0,self.history_dim))
    
        if self.transformer:
            history_nested_list = history_latents_to_nested_list(batch, history_embeds)
            history_nested_list = [torch.cat([t.to(self.device), self.start_token[None]], axis=0) for t in history_nested_list]

            src_padded = nn.utils.rnn.pad_sequence(
                history_nested_list, batch_first=False, padding_value=0
            ) # MAX_LEN x B x D
            
            # The transformer also expects the input to be of type float.
            # src_padded = src_padded.float()
          
            embeddings = self.transformer(src_padded) 
            # embeddings = out.permute(1, 0, 2).squeeze(1)  # history step = 1?? [batch x K x history_dim]
        else:
            embeddings = history_embeds

        # breakpoint()
        # embeddings: MAX_LEN x B x D

        # We may want to put this back into a list.

        if self.repeat_dim == True:  # To point-wise features to concat to DiT
            # breakpoint()
            embeddings = embeddings.unsqueeze(2).repeat(1, 1, self.point_cnts, 1)
        else:
            embeddings = embeddings.unsqueeze(2)
        return embeddings  #


if __name__ == "__main__":
    # Example usage
    # seq_len = 10  # This can vary
    # batch_size = 1
    # input_dim = 7

    # model = HistoryEncoder()
    # src = torch.rand(seq_len, batch_size, input_dim)  # Random input
    # output = model(src)
    # print(output.shape)  # Should be [batch_size, 128]
    import numpy as np
    from torch_geometric.data import Batch, Data

    model = HistoryEncoder(transformer=True)
    data = Data(
        num_points=torch.tensor([1200]),  # N: shape of point cloud
        pos=torch.from_numpy(np.zeros((1200, 3))).float(),
        delta=torch.from_numpy(np.zeros((1200, 3))).float(),
        mask=torch.from_numpy(np.ones(1200)).float(),
        history=torch.from_numpy(np.zeros((1200, 3))).float(),  # N*K, 3
        flow_history=torch.from_numpy(  # N*K, 3
            np.zeros((1200, 3))
        ).float(),  # Snapshot of flow history
        K=1,  # length of history
        lengths=torch.as_tensor([1200]).int(),  # size of point cloud
    )
    batch = Batch.from_data_list([data, data, data])
    history_embed = model(batch)
