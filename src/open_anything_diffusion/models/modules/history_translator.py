# Define the history encoder
import lightning as L
import torch
import torch.nn as nn
import torch_geometric.data as tgd

# from rpad.pyg.nets import pointnet2 as pnp


def get_history_batch(batch):
    """Extracts a single batch of the history data for encoding,
    because each history element is processed separately."""
    history_datas = []
    for data in batch.to_data_list():
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
    return tgd.Batch.from_data_list(history_datas)


def history_latents_to_nested_list(batch, history_latents):
    """Converting history latents from stacked form to nested list"""
    datas = batch.to_data_list()
    history_lengths = [0] + [data.K.item() for data in datas]
    ixs = torch.tensor(history_lengths).cumsum(0).tolist()
    post_encoder_latents = []
    for i, data in enumerate(datas):
        post_encoder_latents.append(history_latents[ixs[i] : ixs[i + 1]])

    return post_encoder_latents


# Previous flow history
class HistoryTranslator(L.LightningModule):
    def __init__(self, history_dim=128, history_len=1, hidden_dim=128):
        super(HistoryTranslator, self).__init__()
        print("Using translator!!!")
        self.history_len = history_len
        assert self.history_len == 1, "currently only supports 1 previous step history"
        self.history_dim = history_dim
        # self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=history_dim)

        # self.history_pcd_encoder = pnp.PN2Dense(
        #     in_channels=3,
        #     out_channels=hidden_dim,
        #     p=pnp.PN2DenseParams(),
        # )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2
        )
        self.output_layer = nn.Linear(
            hidden_dim, history_dim
        )  # Output the same feature dimension as input

    def forward(self, batch):
        point_cnts = batch.lengths[0]

        # Data(
        #     num_points=torch.tensor([1200]), # N: shape of point cloud
        #     pos=torch.from_numpy(np.zeros((1200, 3))).float(),
        #     delta=torch.from_numpy(np.zeros((1200, 3))).float(),
        #     mask=torch.from_numpy(np.ones(1200)).float(),
        #     history=torch.from_numpy(np.zeros((1200, 3))).float(),  # N*K, 3
        #     flow_history=torch.from_numpy( # N*K, 3
        #         np.zeros((1200, 3))
        #     ).float(),  # Snapshot of flow history
        #     K=1,  # length of history
        #     lengths=torch.as_tensor([1200]).int(), # size of point cloud
        # )
        # batch = Batch.from_data_list([data, data, data])
        # history_batch =

        history_pcd_features = self.history_pcd_encoder(batch)

        history_batch = torch.concat(
            [batch.history, history_pcd_features], dim=-1
        ).reshape(-1, point_cnts, 6)
        separate_token = torch.zeros(
            history_batch.shape[0], 1, 6, device=batch.pos.device
        )  # Empty token
        current_batch = torch.concat(
            [batch.pos, torch.zeros_like(batch.pos, device=batch.pos.device)], dim=-1
        ).reshape(-1, point_cnts, 6)
        full_sequence = torch.concat(
            [history_batch, separate_token, current_batch], dim=-2
        )  # bsz * point_size * 128
        # breakpoint()
        transformed = self.transformer_encoder(full_sequence.permute(1, 0, 2))
        # breakpoint()
        embeddings = self.output_layer(transformed).permute(1, 0, 2)[:, -point_cnts:]
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

    model = HistoryTranslator()
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
