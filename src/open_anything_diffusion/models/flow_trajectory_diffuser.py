from typing import Dict

import lightning as L
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch_geometric.data as tgd
from diffusers import DDPMScheduler
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from plotly.subplots import make_subplots
from torch import optim

from open_anything_diffusion.metrics.trajectory import artflownet_loss, flow_metrics


# Flow predictor
class FlowTrajectoryDiffusionModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        # Training params
        self.batch_size = training_cfg.batch_size
        self.lr = training_cfg.lr

        # Diffuser params
        self.sample_size = 1200
        self.time_embed_dim = training_cfg.time_embed_dim
        self.in_channels = 3 * self.traj_len + self.time_embed_dim
        assert (
            network.inchannels == self.in_channels
        ), "Network input channels doesn't match expectation"
        self.traj_len = training_cfg.trajectory_len

        # positional time embeddings
        flip_sin_to_cos = training_cfg.flip_sin_to_cos
        freq_shift = training_cfg.freq_shift
        self.time_proj = Timesteps(64, flip_sin_to_cos, freq_shift)
        timestep_input_dim = training_cfg.time_proj_dim
        self.time_emb = TimestepEmbedding(timestep_input_dim, self.time_embed_dim)

        self.backbone = network
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=training_cfg.config.num_train_timesteps
        )

    def forward(self, data) -> torch.Tensor:  # type: ignore
        timesteps = data.timesteps
        traj_noise = data.traj_noise

        t_emb = self.time_emb(self.time_proj(timesteps))
        # Repeat the time embedding. MAKE SURE THAT EACH BATCH ITEM IS INDEPENDENT!
        t_emb = t_emb.repeat(1, traj_noise.shape[0], 1)  # bs, 1200, 64
        t_emb = torch.flatten(t_emb, start_dim=0, end_dim=1)  # bs * 1200, 64

        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, self.sample_size)

        data.x = torch.cat([traj_noise, t_emb], dim=-1)  # bs * 1200, 64 + 3 * traj_len

        # Run the model.
        pred = self.backbone(data)

        return pred

    def _step(self, batch: tgd.Batch, mode):
        # Make a prediction.
        f_pred = self(batch)
        f_pred = f_pred.reshape(f_pred.shape[0], -1, 3)  # batch * traj_len * 3

        # Compute the loss.
        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point

        f_target = f_target.float()
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log_dict(
            {
                f"{mode}/loss": loss,
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return f_pred, loss

    def predict(self, batch: tgd.Batch, mode):
        bs = batch.delta.shape[0] / self.sample_size
        for t in self.noise_scheduler.timesteps:
            batch.timesteps = torch.zeros(bs) + t  # Uniform t steps
            batch.timesteps = batch.timesteps.long()
            model_output = self(batch)

            # Update traj_noise
            batch.traj_noise = self.noise_scheduler.step(
                model_output, t, batch.traj_noise
            ).prev_sample

        f_pred = batch.traj_noise.transpose(1, 3)
        largest_mag: float = torch.linalg.norm(f_pred, ord=2, dim=-1).max()
        f_pred = f_pred / (largest_mag + 1e-6)

        loss = artflownet_loss(f_pred, batch.delta, self.sample_size)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(
            f_pred[batch.mask == 1], batch.delta[batch.mask == 1]
        )

        self.log_dict(
            {
                f"{mode}/loss": loss,
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return f_pred, loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        # Add timestep & random noise
        bs = batch.delta.shape[0] / self.sample_size
        batch.timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
        ).long()

        noise = torch.randn_like(batch.delta, device=self.device)
        batch.traj_noise = self.noise_scheduler.add_noise(
            batch.delta, noise, batch.timesteps
        )

        _, loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        dataloader_names = ["train", "val", "unseen"]
        name = dataloader_names[dataloader_idx]
        f_pred, loss = self.predict(batch, name)
        return {"preds": f_pred, "loss": loss}

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        obj_id = batch.id
        pos = (
            batch.point[:, -2, :].numpy() if batch.point.shape[1] >= 2 else batch.pos
        )  # The last step's beinning pos
        mask = batch.mask.numpy()
        f_target = batch.delta[:, -1, :]
        f_pred = preds.reshape(preds.shape[0], -1, 3)[:, -1, :]

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

        return {"diffuser_plot": fig}


# class FlowTrajectoryDiffuserInferenceModule(L.LightningModule):
#     def __init__(self, network, inference_config) -> None:
#         super().__init__()
#         self.network = network
#         self.mask_input_channel = inference_config.mask_input_channel

#     def forward(self, data) -> torch.Tensor:  # type: ignore
#         # Maybe add the mask as an input to the network.
#         if self.mask_input_channel:
#             data.x = data.mask.reshape(len(data.mask), 1)

#         # Run the model.
#         trajectory = typing.cast(torch.Tensor, self.network(data))

#         return trajectory

#     def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
#         return self.forward(batch)

#     # the predict step input is different now, pay attention
#     def predict(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """Predict the flow for a single object. The point cloud should
#         come straight from the maniskill processed observation function.

#         Args:
#             xyz (torch.Tensor): Nx3 pointcloud
#             mask (torch.Tensor): Nx1 mask of the part that will move.

#         Returns:
#             torch.Tensor: Nx3 dense flow prediction
#         """
#         print(xyz, mask)
#         assert len(xyz) == len(mask)
#         assert len(xyz.shape) == 2
#         assert len(mask.shape) == 1

#         data = Data(pos=xyz, mask=mask)
#         batch = Batch.from_data_list([data])
#         batch = batch.to(self.device)
#         self.eval()
#         with torch.no_grad():
#             trajectory = self.forward(batch)
#         return trajectory.reshape(trajectory.shape[0], -1, 3)  # batch * traj_len * 3


# class FlowSimulationDiffuserInferenceModule(L.LightningModule):
#     def __init__(self, network) -> None:
#         super().__init__()
#         self.network = network

#     def forward(self, data) -> torch.Tensor:  # type: ignore
#         # Maybe add the mask as an input to the network.
#         rgb, depth, seg, P_cam, P_world, pc_seg, segmap = data

#         data = tgd.Data(
#             pos=torch.from_numpy(P_world).float(),
#             mask=torch.ones(P_world.shape[0]).float(),
#         )
#         batch = tgd.Batch.from_data_list([data])
#         batch = batch.to(self.device)
#         batch.x = batch.mask.reshape(len(batch.mask), 1)
#         self.eval()
#         with torch.no_grad():
#             trajectory = self.network(batch)
#         # print("Trajectory prediction shape:", trajectory.shape)
#         return trajectory.reshape(trajectory.shape[0], -1, 3)
