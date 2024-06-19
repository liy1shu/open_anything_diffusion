# Latent encoding with one history observation (prev PC and flow being encoded)

from typing import Any, Dict

import lightning as L
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch_geometric.data as tgd
from flowbot3d.models.artflownet import artflownet_loss, flow_metrics
from plotly.subplots import make_subplots
from torch import optim

from open_anything_diffusion.metrics.trajectory import (
    artflownet_loss,
    flow_metrics,
    normalize_trajectory,
)

import numpy as np
import math
from torch.nn import init


from online_adaptation.nets.history_nets import *

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


'''
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
       
        # Take the last history observation
        history_data.append(
            tgd.Data(
                x=data.flow_history[-N:],
                pos=data.history[-N:],
            )
        )
        history_datas.extend(history_data)

    return tgd.Batch.from_data_list(history_datas)
'''
# Flow predictor
class FlowHistoryLatentEncodingPredictorTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        in_dim = 0
        p = ArtFlowNetHistoryParams()

        # Latent encoding point net
        self.flownet = PN2DenseLatentEncodingEverywhere(
            p.encoder_dim, in_dim, 3, pnp.PN2DenseParams()
        )


        # Create the history flow encoder
        # Indim is 3 because we are going to pass in the history of prev flows
        self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=p.encoder_dim)
        self.encoder_dim = p.encoder_dim
        self.p = p

        self.no_history_embedding = nn.Parameter(
            torch.randn(p.encoder_dim), requires_grad=True
        )

        self.lr = training_cfg.lr
        self.batch_size = training_cfg.batch_size
        self.mask_input_channel = training_cfg.mask_input_channel

    # Need to unpack the x and pos like how I did in the other file
    def forward(self, batch) -> torch.Tensor:  # type: ignore
        
        # breakpoint()
        history_embeds = torch.zeros(len(batch.lengths), self.encoder_dim).to(
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

        prev_embedding = history_embeds
        
        # history_batch = get_history_batch(batch).to(self.device)
        # prev_embedding = self.prev_flow_encoder(history_batch)

        new_batch = batch.to_data_list()
        for data in new_batch:
            data.pos = data.pos
            data.x = None
        new_batch = tgd.Batch.from_data_list(new_batch).to(self.device)
        pred_flow = self.flownet(new_batch, prev_embedding)

        return pred_flow.unsqueeze(1)

    def _step(self, batch: tgd.Batch, mode):
        # Make a prediction.
        f_pred = self(batch)

        # Compute the loss.
        # breakpoint()
        n_nodes = torch.as_tensor([len(d.mask) for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        f_target = batch.delta.squeeze(1).float()
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log(
            f"{mode}/loss",
            loss,
            add_dataloader_idx=False,
            prog_bar=True,
            batch_size=len(batch),
        )
        self.log_dict(
            {
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )

        return f_pred.reshape(len(batch), -1, 3), loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        f_pred, loss = self._step(batch, "train")
        return {"loss": loss, "preds": f_pred}

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        dataloader_names = ["train", "val", "unseen"]
        name = dataloader_names[dataloader_idx]
        f_pred, loss = self._step(batch, name)
        return {"preds": f_pred, "loss": loss}

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        obj_id = batch.id
        pos = batch.pos.numpy()
        mask = batch.mask.numpy()
        f_target = batch.delta.squeeze(1)
        f_pred = preds.squeeze(0)

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "input data",
                "target flow",
                "pred flow",
            ),
            vertical_spacing=0.05,
        )

        # Parent/child plot.
        labelmap = {0: "unselected", 1: "part"}
        labels = torch.zeros(len(pos)).int()
        labels[mask == 1.0] = 1
        fig.add_traces(v3p._segmentation_traces(pos, labels, labelmap, "scene1"))

        fig.update_layout(
            scene1=v3p._3d_scene(pos),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=1.0, y=0.75),
        )

        # normalize the flow for visualization.
        n_f_gt = (f_target / f_target.norm(dim=1).max()).numpy()
        n_f_pred = (f_pred / f_target.norm(dim=1).max()).numpy()

        # GT flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene2", name="pts"), row=2, col=1)
        f_gt_traces = v3p._flow_traces(
            pos, n_f_gt, scene="scene2", name="f_gt", legendgroup="1"
        )
        fig.add_traces(f_gt_traces, rows=2, cols=1)
        fig.update_layout(scene2=v3p._3d_scene(pos))

        # Predicted flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene3", name="pts"), row=2, col=2)
        f_pred_traces = v3p._flow_traces(
            pos, n_f_pred, scene="scene3", name="f_pred", legendgroup="2"
        )
        fig.add_traces(f_pred_traces, rows=2, cols=2)
        fig.update_layout(scene3=v3p._3d_scene(pos))

        fig.update_layout(title=f"Object {obj_id}")

        return {"artflownet_plot": fig}



class FlowTrajectoryDiffuserSimulationModule_latent_encoding(L.LightningModule):
    def __init__(self, network, inference_config, model_config) -> None:
        super().__init__()
        
        in_dim = 0
        p = ArtFlowNetHistoryParams()

        # Latent encoding point net
        # self.flownet = PN2DenseLatentEncodingEverywhere(
        #     p.encoder_dim, in_dim, 3, pnp.PN2DenseParams()
        # )
        self.model = FlowHistoryLatentEncodingPredictorInferenceModule(network, inference_config)

        # Create the history flow encoder
        # Indim is 3 because we are going to pass in the history of prev flows
        self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=p.encoder_dim)

        # Start token
        # self.start_token = nn.Parameter(torch.empty((p.encoder_dim,)), requires_grad=True)
        # bound = 1/math.sqrt(p.encoder_dim)
        # init.uniform_(self.start_token, -bound, bound)
        self.no_history_embedding = nn.Parameter(
            torch.randn(p.encoder_dim), requires_grad=True
        )

        self.p = p

        self.batch_size = inference_config.batch_size
        # self.mask_input_ch
        self.history_len = 1
        self.sample_size = 1200

    def load_from_ckpt(self, ckpt_file):
        # self.model.flownet.load_from_ckpt(ckpt_file)
        ckpt = torch.load(ckpt_file)
        # breakpoint()
        self.model.load_state_dict(ckpt["state_dict"])

    def forward(self, data, history_pcd=None, history_flow=None) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = data
        K = self.history_len
        if history_pcd is None:
            history_pcd = np.zeros_like(P_world)
            history_flow = np.zeros_like(P_world)
            K = 0
        data = tgd.Data(
            pos=torch.from_numpy(P_world).float().cuda(),
            history=torch.from_numpy(history_pcd).float().cuda(),
            flow_history=torch.from_numpy(history_flow).float().cuda(),
            K=K,
            lengths=self.sample_size
            # mask=torch.ones(P_world.shape[0]).float(),
        )
        # breakpoint()
        batch = tgd.Batch.from_data_list([data])
        # batch = batch.to(self.device)
        # batch.x = batch.mask.reshape(len(batch.mask), 1)
        self.eval()
        with torch.no_grad():
            # trajectory = self.model.faster_predict_step(batch, 0)
            trajectory = self.model.predict_step(batch, 0)
        # print("Trajectory prediction shape:", trajectory.shape)
        return trajectory.cpu()


class FlowHistoryLatentEncodingPredictorInferenceModule(L.LightningModule):
    def __init__(self, network, inference_config) -> None:
        super().__init__()

        in_dim = 0
        p = ArtFlowNetHistoryParams()

        # Latent encoding point net
        self.flownet = PN2DenseLatentEncodingEverywhere(
            p.encoder_dim, in_dim, 3, pnp.PN2DenseParams()
        )

        # Create the history flow encoder
        # Indim is 3 because we are going to pass in the history of prev flows
        self.prev_flow_encoder = pnp.PN2Encoder(in_dim=3, out_dim=p.encoder_dim)

        # Start token
        # self.start_token = nn.Parameter(torch.empty((p.encoder_dim,)), requires_grad=True)
        # bound = 1/math.sqrt(p.encoder_dim)
        # init.uniform_(self.start_token, -bound, bound)
        self.no_history_embedding = nn.Parameter(
            torch.randn(p.encoder_dim), requires_grad=True
        )

        self.p = p

        self.batch_size = inference_config.batch_size
        # self.mask_input_channel = inference_config.mask_input_channel

        # self.vectorize = inference_config.vectorize
        self.history_len = 1
        self.sample_size = 1200
        self.traj_len = inference_config.trajectory_len
        self.encoder_dim = p.encoder_dim


    def load_from_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        # breakpoint()
        self.flownet.load_state_dict(ckpt["state_dict"])

    # copy over the training module after it works
    def forward(self, batch) -> torch.Tensor:  # type: ignore
        # breakpoint()
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = batch # should use this to split the batch
        history_embeds = torch.zeros(len(batch.lengths), self.encoder_dim).to(
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

        prev_embedding = history_embeds
        # history_batch = get_history_batch(batch).to(self.device)
        # prev_embedding = self.prev_flow_encoder(history_batch)
        # # put learnable start token here

        new_batch = batch.to_data_list()
        for data in new_batch:
            data.pos = data.pos
            data.x = None
        new_batch = tgd.Batch.from_data_list(new_batch).to(self.device)

        pred_flow = self.flownet(new_batch, prev_embedding)

        return pred_flow

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        # return self.forward(batch)
        # breakpoint()
        # torch.eval()
        self.eval()
        # bs = batch.pos.shape[0] // self.sample_size
        # z = torch.randn(bs, 3 * self.traj_len, 30, 40, device=self.device)  # .float()

        # history_embed = (
        #     self.prev_flow_encoder(batch).permute(0, 2, 1).squeeze(-1)
        # )  # History embedding
        # batch.history_embed = history_embed
        # pos = (
        #     batch.pos.reshape(-1, self.sample_size, 3 * self.traj_len)
        #     .permute(0, 2, 1)
        #     .float()
        #     .cuda()
        # )
        # model_kwargs = dict(pos=pos, context=batch.cuda())

        # samples, results = self.diffusion.p_sample_loop(
        #     self.backbone,
        #     z.shape,
        #     z,
        #     clip_denoised=False,
        #     model_kwargs=model_kwargs,
        #     progress=True,
        #     device=self.device,
        # )

        # f_pred = (
        #     torch.flatten(samples, start_dim=2, end_dim=3)
        #     .permute(0, 2, 1)
        #     .reshape(-1, 3 * self.traj_len)
        #     .unsqueeze(1)
        # )
        f_pred = self.forward(batch).unsqueeze(1)
        f_pred = normalize_trajectory(f_pred)
        return f_pred

    # the predict step input is different now, pay attention
    def predict(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict the flow for a single object. The point cloud should
        come straight from the maniskill processed observation function.

        Args:
            xyz (torch.Tensor): Nx3 pointcloud
            mask (torch.Tensor): Nx1 mask of the part that will move.

        Returns:
            torch.Tensor: Nx3 dense flow prediction
        """
        print(xyz, mask)
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        data = Data(pos=xyz, mask=mask)
        batch = tgd.Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            flow = self.forward(batch)
        return flow
