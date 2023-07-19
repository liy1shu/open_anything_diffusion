import typing
from typing import Dict

import lightning as L
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch_geometric.data as tgd
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation

# from flowbot3d.models.artflownet import artflownet_loss, flow_metrics
from flowbot3d.models.artflownet import artflownet_loss
from plotly.subplots import make_subplots
from torch import optim


def flow_metrics(pred_flow, gt_flow):
    with torch.no_grad():
        # RMSE
        rmse = (pred_flow - gt_flow).norm(p=2, dim=-1).mean()

        # Cosine similarity, normalized.
        nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=-1) != 0.0)
        gt_flow_nz = gt_flow[nonzero_gt_flowixs]
        pred_flow_nz = pred_flow[nonzero_gt_flowixs]
        cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=-1).mean()

        # Magnitude
        mag_error = (
            (pred_flow.norm(p=2, dim=-1) - gt_flow.norm(p=2, dim=-1)).abs().mean()
        )
    return rmse, cos_dist, mag_error


def make_trajectory_animation(traj_data):  # Make trajectory animation
    animation = FlowNetAnimation()
    for i in range(traj_data["point"].shape[-2]):
        pcd = (
            traj_data["pos"].detach().numpy()
            if i == 0
            else traj_data["point"][:, (i - 1)].detach().numpy()
        )
        mask = traj_data["mask"].detach().numpy()
        flow = traj_data["delta"][:, i].detach().numpy()
        animation.add_trace(
            torch.as_tensor(pcd),
            torch.as_tensor([pcd]),
            torch.as_tensor([flow]),
            "red",
        )

    return animation.animate()


# Flow predictor
class FlowTrajectoryTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        self.network = network
        self.lr = training_cfg.lr
        self.batch_size = training_cfg.batch_size
        self.mask_input_channel = training_cfg.mask_input_channel
        self.mode = training_cfg.mode
        self.trajectory_len = training_cfg.trajectory_len
        assert self.mode in ["delta", "point"]

    def forward(self, data) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        if self.mask_input_channel:
            data.x = data.mask.reshape(len(data.mask), 1)

        # Run the model.
        flow = typing.cast(torch.Tensor, self.network(data))

        return flow

    def _step(self, batch: tgd.Batch, mode):
        # Make a prediction.
        f_pred = self(batch)

        # Compute the loss.
        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta.reshape(batch.delta.shape[0], -1)
        elif self.mode == "point":
            f_target = batch.point.reshape(batch.point.shape[0], -1)

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
        return f_pred.reshape(f_pred.shape[0], -1, 3), loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        _, loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        dataloader_names = ["train", "val", "unseen"]
        name = dataloader_names[dataloader_idx]
        f_pred, loss = self._step(batch, name)
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

        return {"artflownet_plot": fig}


class FlowPredictorInferenceModule(L.LightningModule):
    def __init__(self, network, inference_config) -> None:
        super().__init__()
        self.network = network
        self.mask_input_channel = inference_config.mask_input_channel

    def forward(self, data) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        if self.mask_input_channel:
            data.x = data.mask.reshape(len(data.mask), 1)

        # Run the model.
        flow = typing.cast(torch.Tensor, self.network(data))

        return flow

    # TODO: the predict step input is different now, pay attention
    def predict_step(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict the flow for a single object. The point cloud should
        come straight from the maniskill processed observation function.

        Args:
            xyz (torch.Tensor): Nx3 pointcloud
            mask (torch.Tensor): Nx1 mask of the part that will move.

        Returns:
            torch.Tensor: Nx3 dense flow prediction
        """
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        data = Data(pos=xyz, mask=mask)
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            flow = self.forward(batch)
        return flow
