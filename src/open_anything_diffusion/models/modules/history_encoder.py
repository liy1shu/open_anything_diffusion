# Define the history encoder
import lightning as L
import torch
import torch.nn as nn
import torch_geometric.data as tgd


# Yishu's old old old version - history : grasp point & direction & outcome
class PDOHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=32,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        batch_norm=False,
    ):
        super(PDOHistoryEncoder, self).__init__()
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


def get_history_batch(batch):
    """Extracts a single batch of the history data for encoding,
    because each history element is processed separately."""
    has_history_ids = []
    no_history_ids = []
    history_datas = []
    for id, data in enumerate(batch.to_data_list()):
        if data.K == 0:  # No history
            no_history_ids.append(id)
            continue
        has_history_ids.append(id)
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
    if len(history_datas) == 0:
        return no_history_ids, has_history_ids, None  # No has_history batch
    return no_history_ids, has_history_ids, tgd.Batch.from_data_list(history_datas)


def history_latents_to_nested_list(batch, history_latents):
    """Converting history latents from stacked form to nested list"""
    datas = batch.to_data_list()
    # history_lengths = [0] + [data.K.item() for data in datas]
    history_lengths = [0] + [1 for data in datas]
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
        if self.transformer:
            self.transformer = nn.Transformer(d_model=history_dim)

    def forward(self, batch):
        # point_cnts = batch.lengths[0]

        history_embeds = torch.zeros(len(batch.lengths), self.history_dim).to(
            self.device
        )  # Also add the no history batch
        no_history_ids, has_history_ids, history_batch = get_history_batch(batch)
        # print("bsz = ", len(batch.lengths))
        if len(has_history_ids) != 0:  # Has history samples
            history_batch = history_batch.to(self.device)
            has_history_embeds = self.prev_flow_encoder(history_batch)
            history_embeds[has_history_ids] += has_history_embeds
        if len(no_history_ids) != 0:  # Has no history samples
            history_embeds[no_history_ids] += self.no_history_embedding

        if self.transformer:
            history_nested_list = history_latents_to_nested_list(batch, history_embeds)
            src_padded = nn.utils.rnn.pad_sequence(
                history_nested_list, batch_first=False, padding_value=0
            )
            # Create a mask for the padded sequences.
            src_mask = (src_padded == 0.0).all(dim=-1)

            # This is our query vector. It has shape [S, N, E], where S is the sequence length, N is the batch size, and E is the embedding size.
            tgt = torch.ones(1, batch.num_graphs, self.history_dim).to(self.device)

            # The transformer also expects the input to be of type float.
            src_padded = src_padded.float()
            tgt = tgt.float()
            # Pass the input through the transformer, with mask and tgt.
            out = self.transformer(
                src_padded, tgt, src_key_padding_mask=src_mask.transpose(1, 0)
            )

            embeddings = out.permute(1, 0, 2).squeeze(1)  # history step = 1
        else:
            embeddings = history_embeds

        if self.repeat_dim == True:  # To point-wise features to concat to DiT
            embeddings = embeddings.unsqueeze(1).repeat(1, self.point_cnts, 1)
        else:
            embeddings = embeddings.unsqueeze(1)
        return embeddings


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
