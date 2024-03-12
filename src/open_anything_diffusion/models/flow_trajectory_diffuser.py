from typing import Any, Dict

import lightning as L
import plotly.express as px
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch.nn.functional as F
import torch_geometric.data as tgd
import tqdm
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.optimization import get_cosine_schedule_with_warmup
from plotly.subplots import make_subplots
from torch import optim

from open_anything_diffusion.metrics.trajectory import (
    artflownet_loss,
    flow_metrics,
    normalize_trajectory,
)


# Flow diffuser with PN++
class FlowTrajectoryDiffusionModule_PN2(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg) -> None:
        super().__init__()
        # Training params
        self.batch_size = training_cfg.batch_size
        self.lr = training_cfg.lr
        self.mode = training_cfg.mode
        self.traj_len = training_cfg.trajectory_len
        self.epochs = training_cfg.epochs
        self.train_sample_number = training_cfg.train_sample_number

        # Diffuser training param
        self.wta = training_cfg.wta
        self.wta_trial_times = training_cfg.wta_trial_times
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # Diffuser params
        self.sample_size = 1200
        self.time_embed_dim = model_cfg.time_embed_dim
        self.in_channels = 3 * self.traj_len + self.time_embed_dim
        assert (
            network.in_ch == self.in_channels
        ), "Network input channels doesn't match expectation"

        # positional time embeddings
        flip_sin_to_cos = model_cfg.flip_sin_to_cos
        freq_shift = model_cfg.freq_shift
        self.time_proj = Timesteps(64, flip_sin_to_cos, freq_shift)
        timestep_input_dim = model_cfg.time_proj_dim
        self.time_emb = TimestepEmbedding(timestep_input_dim, self.time_embed_dim)

        self.backbone = network
        self.num_train_timesteps = model_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_cfg.num_train_timesteps
        )

        self.cosine_distribution_cache = {"x": [], "y": [], "colors": []}

    def forward(self, data) -> torch.Tensor:  # type: ignore
        timesteps = data.timesteps
        traj_noise = data.traj_noise  # bs * 1200, traj_len, 3
        traj_noise = torch.flatten(traj_noise, start_dim=1, end_dim=2)

        t_emb = self.time_emb(self.time_proj(timesteps))  # bs, 64
        # Repeat the time embedding. MAKE SURE THAT EACH BATCH ITEM IS INDEPENDENT!
        t_emb = t_emb.unsqueeze(1).repeat(1, self.sample_size, 1)  # bs, 1200, 64
        t_emb = torch.flatten(t_emb, start_dim=0, end_dim=1)  # bs * 1200, 64

        data.x = torch.cat([traj_noise, t_emb], dim=-1)  # bs * 1200, 64 + 3 * traj_len

        # Run the model.
        pred = self.backbone(data)

        return pred

    def _step(self, batch: tgd.Batch, mode):
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point
        f_target = f_target  # .float()

        # noisy_flow = batch.traj_noise
        # Predict the noise.
        noise = batch.pure_noise  # The added noise
        noise_pred = self(batch)
        noise_pred = noise_pred.reshape(
            noise_pred.shape[0], -1, 3
        )  # batch, traj_len, 3

        # Compute the loss.
        loss = F.mse_loss(noise_pred, noise)

        self.log_dict(
            {
                f"{mode}/loss": loss,
                # The other metrics will be tested in validation
                # f"{mode}/rmse": rmse,
                # f"{mode}/cosine_similarity": cos_dist,
                # f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return None, loss

    # @torch.inference_mode()
    def predict(self, batch: tgd.Batch, mode):
        # torch.eval()
        bs = batch.delta.shape[0] // self.sample_size
        batch.traj_noise = torch.randn_like(batch.delta, device=self.device)  # .float()
        # batch.traj_noise = normalize_trajectory(batch.traj_noise)
        # breakpoint()

        # import time
        # batch_time = 0
        # model_time = 0
        # noise_scheduler_time = 0

        # self.noise_scheduler_inference.set_timesteps(self.num_inference_timesteps)
        # print(self.noise_scheduler_inference.timesteps)
        # for t in self.noise_scheduler_inference.timesteps:
        for t in self.noise_scheduler.timesteps:
            # tm = time.time()
            batch.timesteps = torch.zeros(bs, device=self.device) + t  # Uniform t steps
            batch.timesteps = batch.timesteps.long()
            # batch_time += time.time() - tm

            # tm = time.time()
            model_output = self(batch)  # bs * 1200, traj_len * 3
            model_output = model_output.reshape(
                model_output.shape[0], -1, 3
            )  # bs * 1200, traj_len, 3

            batch.traj_noise = self.noise_scheduler.step(
                # batch.traj_noise = self.noise_scheduler_inference.step(
                model_output.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
                t,
                batch.traj_noise.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
            ).prev_sample
            batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        f_pred = batch.traj_noise  # .float()
        f_pred = normalize_trajectory(f_pred)

        # Compute the loss.
        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point

        f_target = f_target  # .float()
        f_target = normalize_trajectory(f_target)
        # print(f_pred[f_ix], batch.delta[f_ix])
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log_dict(
            {
                f"{mode}/flow_loss": loss,
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return f_pred, loss

    def predict_wta(self, orig_batch: tgd.Batch, mode):
        bs = orig_batch.delta.shape[0] // self.sample_size
        assert bs == 1, f"batch size should be 1, now is {bs}"

        # batch every sample into bsz of trial_times
        bs = self.wta_trial_times
        data_list = orig_batch.to_data_list() * self.wta_trial_times
        batch = tgd.Batch.from_data_list(data_list)

        batch.traj_noise = torch.randn_like(batch.delta, device=self.device)

        for t in self.noise_scheduler.timesteps:
            batch.timesteps = torch.zeros(bs, device=self.device) + t  # Uniform t steps
            batch.timesteps = batch.timesteps.long()

            model_output = self(batch)  # bs * 1200, traj_len * 3
            model_output = model_output.reshape(
                model_output.shape[0], -1, 3
            )  # bs * 1200, traj_len, 3

            batch.traj_noise = self.noise_scheduler.step(
                model_output.reshape(
                    -1,
                    self.sample_size,
                    model_output.shape[1],
                    model_output.shape[2],
                ),
                t,
                batch.traj_noise.reshape(
                    -1,
                    self.sample_size,
                    model_output.shape[1],
                    model_output.shape[2],
                ),
            ).prev_sample
            batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        f_pred = batch.traj_noise  # .float()
        f_pred = normalize_trajectory(f_pred)

        # Compute the loss.
        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point

        f_target = f_target  # .float()
        f_target = normalize_trajectory(f_target)
        flow_loss = artflownet_loss(f_pred, f_target, n_nodes, reduce=False)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(
            f_pred[f_ix], f_target[f_ix], reduce=False
        )

        # Aggregate the results
        # Choose the one with smallest flow loss
        flow_loss = flow_loss.reshape(bs, -1).mean(-1)
        rmse = rmse.reshape(bs, -1).mean(-1)
        cos_dist = cos_dist.reshape(bs, -1).mean(-1)
        mag_error = mag_error.reshape(bs, -1).mean(-1)

        chosen_id = torch.min(flow_loss, 0)[1]  # index
        pos_cosine = torch.sum((cos_dist - 0.7) > 0) / bs
        neg_cosine = torch.sum((cos_dist + 0.7) < 0) / bs
        multimodal = 1 if (pos_cosine != 0 and neg_cosine != 0) else 0

        self.log_dict(
            {
                f"{mode}_wta/flow_loss": flow_loss[chosen_id].item(),
                f"{mode}_wta/rmse": rmse[chosen_id].item(),
                f"{mode}_wta/cosine_similarity": cos_dist[chosen_id].item(),
                f"{mode}_wta/mag_error": mag_error[chosen_id].item(),
                f"{mode}_wta/multimodal": multimodal,
                f"{mode}_wta/pos@0.7": pos_cosine.item(),
                f"{mode}_wta/neg@0.7": neg_cosine.item(),
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return (
            f_pred.reshape(bs, self.sample_size, self.traj_len, 3)[chosen_id],
            flow_loss[chosen_id],
            cos_dist.tolist(),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(
                (self.train_sample_number // self.batch_size) * self.epochs
            ),
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        batch.delta = normalize_trajectory(batch.delta)
        # Add timestep & random noise
        bs = batch.delta.shape[0] // self.sample_size
        # breakpoint()
        batch.timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
        ).long()

        noise = torch.randn_like(batch.delta, device=self.device)
        batch.pure_noise = noise
        # broadcast_timestep = torch.flatten(batch.timesteps.unsqueeze(1).repeat(1, self.sample_size), start_dim=0, end_dim=-1)  # bs * 1200

        batch.traj_noise = self.noise_scheduler.add_noise(
            batch.delta.reshape(-1, self.sample_size, noise.shape[1], noise.shape[2]),
            noise.reshape(
                -1, self.sample_size, noise.shape[1], noise.shape[2]
            ),  # bs, 1200, traj_len, 3
            # broadcast_timestep
            batch.timesteps,
        )
        batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        _, loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()

        # Clean cache for a new eval dataloader
        if batch_id == 0:
            self.cosine_distribution_cache["x"] = []
            self.cosine_distribution_cache["y"] = []
            self.cosine_distribution_cache["colors"] = []

        dataloader_names = ["val", "train", "unseen"]
        name = dataloader_names[dataloader_idx]
        with torch.no_grad():
            f_pred, loss = self.predict(batch, name)
            if self.wta:
                f_pred, loss, cosines = self.predict_wta(batch, name)
                self.cosine_distribution_cache["x"] += [batch_id] * self.wta_trial_times
                self.cosine_distribution_cache["y"] += cosines
                self.cosine_distribution_cache["colors"] += [
                    "blue" if batch_id % 2 == 0 else "red"
                ] * self.wta_trial_times
        # breakpoint()
        return {
            "preds": f_pred,
            "loss": loss,
            "cosine_cache": self.cosine_distribution_cache,
        }

    @staticmethod
    def make_plots(preds, batch: tgd.Batch, cosine_cache=None) -> Dict[str, go.Figure]:
        # 1) Make the flow visualization plots
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

        # 2) Make the cosine distribution plots
        cos_fig = None
        if (
            cosine_cache is not None and len(cosine_cache["x"]) != 0
        ):  # Does wta, and needs plots
            # The following Matplotlib code won't work because some matplotlib version issue (3.4.3 would work, but the version is old)
            # cos_fig = plt.figure()
            # ax = cos_fig.add_axes([0.1, 0.1, 0.8, 0.8])
            # plt.ylim((-1, 1))
            # ax.axhline(y=0)
            # ax.axhline(y=0.7)
            # ax.axhline(y=-0.7)
            # plt.scatter(cosine_cache["x"], cosine_cache["y"], s=5, c=cosine_cache["colors"])
            cos_fig = px.scatter(
                x=cosine_cache["x"], y=cosine_cache["y"], color=cosine_cache["colors"]
            )
            cos_fig.update_layout(yaxis_range=[-1, 1])

        return {"diffuser_plot": fig, "cosine_distribution_plot": cos_fig}


class FlowTrajectoryDiffuserInferenceModule_PN2(L.LightningModule):
    def __init__(self, network, inference_cfg, model_cfg) -> None:
        super().__init__()
        # Inference params
        self.batch_size = inference_cfg.batch_size
        self.traj_len = inference_cfg.trajectory_len

        # Diffuser params
        self.sample_size = 1200
        self.time_embed_dim = model_cfg.time_embed_dim
        self.in_channels = 3 * self.traj_len + self.time_embed_dim
        assert (
            network.in_ch == self.in_channels
        ), "Network input channels doesn't match expectation"

        # positional time embeddings
        flip_sin_to_cos = model_cfg.flip_sin_to_cos
        freq_shift = model_cfg.freq_shift
        self.time_proj = Timesteps(64, flip_sin_to_cos, freq_shift)
        timestep_input_dim = model_cfg.time_proj_dim
        self.time_emb = TimestepEmbedding(timestep_input_dim, self.time_embed_dim)

        self.backbone = network
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_cfg.num_train_timesteps
        )
        self.faster_noise_scheduler = DDIMScheduler(
            num_train_timesteps=model_cfg.num_train_timesteps
        )

    def load_from_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.load_state_dict(ckpt["state_dict"])

    def forward(self, data) -> torch.Tensor:  # type: ignore
        timesteps = data.timesteps
        traj_noise = data.traj_noise  # bs * 1200, traj_len, 3
        traj_noise = torch.flatten(traj_noise, start_dim=1, end_dim=2)

        t_emb = self.time_emb(self.time_proj(timesteps))  # bs, 64
        # Repeat the time embedding. MAKE SURE THAT EACH BATCH ITEM IS INDEPENDENT!
        t_emb = t_emb.unsqueeze(1).repeat(1, self.sample_size, 1)  # bs, 1200, 64
        t_emb = torch.flatten(t_emb, start_dim=0, end_dim=1)  # bs * 1200, 64

        data.x = torch.cat([traj_noise, t_emb], dim=-1)  # bs * 1200, 64 + 3 * traj_len
        # Run the model.
        data.mask = data.mask.to(self.device)
        data.pos = data.pos.to(self.device)
        data.ptr = data.ptr.to(self.device)
        data.batch = data.batch.to(self.device)
        pred = self.backbone(data)

        return pred

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        bs = batch.pos.shape[0] // self.sample_size
        batch.traj_noise = torch.randn(
            (batch.pos.shape[0], self.traj_len, 3), device=self.device
        )  # .float()
        for t in self.noise_scheduler.timesteps:
            batch.timesteps = torch.zeros(bs, device=self.device) + t  # Uniform t steps
            batch.timesteps = batch.timesteps.long()
            model_output = self(batch)  # bs * 1200, traj_len * 3
            model_output = model_output.reshape(
                model_output.shape[0], -1, 3
            )  # bs * 1200, traj_len, 3

            batch.traj_noise = self.noise_scheduler.step(
                # batch.traj_noise = self.noise_scheduler_inference.step(
                model_output.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
                t,
                batch.traj_noise.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
            ).prev_sample
            batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        f_pred = batch.traj_noise  # .float()

        # print(f_pred.shape)
        return f_pred

    @torch.no_grad()
    def faster_predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        bs = batch.pos.shape[0] // self.sample_size
        batch.traj_noise = torch.randn(
            (batch.pos.shape[0], self.traj_len, 3), device=self.device
        )  # .float()
        self.faster_noise_scheduler.set_timesteps(100)
        for t in self.noise_scheduler.timesteps:
            batch.timesteps = torch.zeros(bs, device=self.device) + t  # Uniform t steps
            batch.timesteps = batch.timesteps.long()
            model_output = self(batch)  # bs * 1200, traj_len * 3
            model_output = model_output.reshape(
                model_output.shape[0], -1, 3
            )  # bs * 1200, traj_len, 3

            batch.traj_noise = self.faster_noise_scheduler.step(
                # batch.traj_noise = self.noise_scheduler_inference.step(
                model_output.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
                t,
                batch.traj_noise.reshape(
                    -1, self.sample_size, model_output.shape[1], model_output.shape[2]
                ),
            ).prev_sample
            batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        f_pred = batch.traj_noise  # .float()

        print(f_pred.shape)
        return f_pred

    # For winner takes it all evaluation
    @torch.inference_mode()
    def predict_wta(self, dataloader, mode="delta", trial_times=50):
        print(self.device)
        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        all_flow_loss = 0
        all_multimodal = 0
        all_pos_cosine = 0
        all_neg_cosine = 0

        for id, orig_sample in tqdm.tqdm(enumerate(dataloader)):
            bs = orig_sample.delta.shape[0] // self.sample_size
            assert bs == 1, f"batch size should be 1, now is {bs}"

            # batch every sample into bsz of trial_times
            data_list = orig_sample.to_data_list() * trial_times
            batch = tgd.Batch.from_data_list(data_list)
            bs = trial_times

            batch.traj_noise = torch.randn_like(batch.delta, device=self.device).to(
                self.device
            )  # .float()

            for t in self.noise_scheduler.timesteps:
                batch.timesteps = (
                    torch.zeros(bs, device=self.device) + t
                )  # Uniform t steps
                batch.timesteps = batch.timesteps.long()

                model_output = self(batch)  # bs * 1200, traj_len * 3
                model_output = model_output.reshape(
                    model_output.shape[0], -1, 3
                )  # bs * 1200, traj_len, 3

                batch.traj_noise = self.noise_scheduler.step(
                    model_output.reshape(
                        -1,
                        self.sample_size,
                        model_output.shape[1],
                        model_output.shape[2],
                    ),
                    t,
                    batch.traj_noise.reshape(
                        -1,
                        self.sample_size,
                        model_output.shape[1],
                        model_output.shape[2],
                    ),
                ).prev_sample
                batch.traj_noise = torch.flatten(
                    batch.traj_noise, start_dim=0, end_dim=1
                )

            f_pred = batch.traj_noise  # .float()
            f_pred = normalize_trajectory(f_pred)

            # Compute the loss.
            n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
            f_ix = batch.mask.bool().to(self.device)
            if mode == "delta":
                f_target = batch.delta.to(self.device)
            elif mode == "point":
                f_target = batch.point.to(self.device)

            f_target = f_target  # .float()
            f_target = normalize_trajectory(f_target)
            flow_loss = artflownet_loss(f_pred, f_target, n_nodes, reduce=False)

            # Compute some metrics on flow-only regions.
            rmse, cos_dist, mag_error = flow_metrics(
                f_pred[f_ix], f_target[f_ix], reduce=False
            )

            # Aggregate the results
            # Choose the one with smallest flow loss
            flow_loss = flow_loss.reshape(bs, -1).mean(-1)
            rmse = rmse.reshape(bs, -1).mean(-1)
            cos_dist = cos_dist.reshape(bs, -1).mean(-1)

            # all_directions += list(cos_dist)

            mag_error = mag_error.reshape(bs, -1).mean(-1)

            chosen_id = torch.min(flow_loss, 0)[1]  # index
            pos_cosine = torch.sum((cos_dist - 0.7) > 0) / bs
            neg_cosine = torch.sum((cos_dist + 0.7) < 0) / bs
            multimodal = 1 if (pos_cosine != 0 and neg_cosine != 0) else 0

            print(
                multimodal,
                rmse[chosen_id],
                cos_dist[chosen_id],
                mag_error[chosen_id],
                flow_loss[chosen_id],
            )
            all_multimodal += multimodal  # .item()
            all_rmse += rmse[chosen_id].item()
            all_cos_dist += cos_dist[chosen_id].item()
            all_mag_error += mag_error[chosen_id].item()
            all_flow_loss += flow_loss[chosen_id].item()
            all_pos_cosine += pos_cosine.item()
            all_neg_cosine += neg_cosine.item()

        metric_dict = {
            f"flow_loss": all_flow_loss / len(dataloader),
            f"rmse": all_rmse / len(dataloader),
            f"cosine_similarity": all_cos_dist / len(dataloader),
            f"mag_error": all_mag_error / len(dataloader),
            f"multimodal": all_multimodal / len(dataloader),
            f"pos@0.7": all_pos_cosine / len(dataloader),
            f"neg@0.7": all_neg_cosine / len(dataloader),
        }

        self.log_dict(
            metric_dict,
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return metric_dict, cos_dist.tolist()  # dataloader * trial_times


class FlowTrajectoryDiffuserSimulationModule_PN2(L.LightningModule):
    def __init__(self, network, inference_cfg, model_cfg) -> None:
        super().__init__()
        self.model = FlowTrajectoryDiffuserInferenceModule_PN2(
            network, inference_cfg, model_cfg
        )

    def load_from_ckpt(self, ckpt_file):
        self.model.load_from_ckpt(ckpt_file)

    def forward(self, data) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = data

        data = tgd.Data(
            pos=torch.from_numpy(P_world).float().cuda(),
            # mask=torch.ones(P_world.shape[0]).float(),
        )
        batch = tgd.Batch.from_data_list([data])
        # batch = batch.to(self.device)
        # batch.x = batch.mask.reshape(len(batch.mask), 1)
        self.eval()
        with torch.no_grad():
            # trajectory = self.model.faster_predict_step(batch, 0)
            trajectory = self.model.predict_step(batch, 0)
        # print("Trajectory prediction shape:", trajectory.shape)
        return trajectory.cpu()
