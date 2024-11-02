## Diffusion model
from dataclasses import dataclass

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.data as tgd

# import rpad.pyg.nets.pointnet2 as pnp
import tqdm
import wandb
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation

from open_anything_diffusion.datasets.flow_history_pdo import FlowHistoryDataModule
from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule
from open_anything_diffusion.metrics.trajectory import (
    artflownet_loss,
    flow_metrics,
    normalize_trajectory,
)
from open_anything_diffusion.models.diffusion.history_model import PNHistoryDiffuser
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder

# import open_anything_diffusion.models.diffusion.module as pnp


@dataclass
class TrainingConfig:
    device = "cuda"

    # batch_size = 32
    batch_size = 1
    # train_batch_size = 16
    # eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 400
    # num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    train_sample_number = 1

    traj_len = 1
    # Diffuser params
    num_train_timesteps = 100
    seed = 0
    sample_size = 1200
    in_channels = 3
    out_channels = 3
    cross_attention_dim = 3
    block_out_channels = [128, 256, 512, 512]
    attention_head_dim = 3

    # History encoder params
    history_input_dim = 7
    history_embed_dim = 128
    # history_d_model = 128
    # nhead = 4
    # num_layers = 2
    # dim_feedforward = 256

    # ckpt params
    read_ckpt_path = "./diffusion_condition_ckpt.pth"
    save_ckpt_path = "./diffusion_condition_ckpt.pth"


class HistoryTrajDiffuser:
    def __init__(self, config, train_batch_num):
        self.config = config
        self.traj_len = config.traj_len
        self.device = config.device

        self.history_encoder = HistoryEncoder(
            input_dim=config.history_input_dim, output_dim=config.history_embed_dim
        ).to(config.device)

        self.model = PNHistoryDiffuser(
            in_channels=3 * config.traj_len,
            # sample_size=1200,
            traj_len=config.traj_len,
            time_embed_dim=64,
            history_embed_dim=config.history_embed_dim
            # emb_dims=3
        ).to(config.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps
        )
        self.optimizer_diffuser = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self.optimizer_history = torch.optim.Adam(
            self.history_encoder.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.lr_scheduler_diffuser = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer_diffuser,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(train_batch_num * config.num_epochs),
            # num_training_steps=((config.train_sample_number // config.batch_size) * config.num_epochs),
            # num_training_steps=(config.num_epochs),
        )

    def load_model(self, ckpt_path="./diffusion_best_ckpt.pth"):
        self.model.load_state_dict(torch.load(ckpt_path))

    def train(
        self, train_dataloader, train_val_dataloader, val_dataloader, unseen_dataloader
    ):
        # Train loop
        losses = []
        min_loss = 10
        global_step = 0
        for epoch in range(config.num_epochs):
            print(f"Epoch: {epoch}")
            self.model.train()
            self.history_encoder.train()

            accu_loss = torch.tensor(0).float().to(self.device)
            for id, batch in tqdm.tqdm(enumerate(train_dataloader)):
                # The history embedding
                history = torch.cat(
                    [
                        batch.trial_points,
                        batch.trial_directions,
                        batch.trial_results.unsqueeze(-1),
                    ],
                    dim=-1,
                ).unsqueeze(0)
                history_embed = self.history_encoder(history.to(self.device))
                batch.history_embed = history_embed.repeat(self.config.sample_size, 1)

                global_step += 1

                clean_flow = batch.delta
                clean_flow = clean_flow.reshape(
                    -1, 1200, clean_flow.shape[1], clean_flow.shape[2]
                ).to(self.device)
                condition = batch.pos
                condition = condition.reshape(-1, 1200, condition.shape[1]).to(
                    self.device
                )
                clean_flow = clean_flow.transpose(1, 3)

                # Sample noise to add
                noise = torch.randn(clean_flow.shape).to(clean_flow.device)
                bs = clean_flow.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_flow.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(
                    clean_flow, noise, timesteps
                )

                # Predict the noise residual
                noise_pred = self.model(
                    noisy_images, timesteps, context=batch, return_dict=False
                )[0]
                loss = F.mse_loss(noise_pred, noise)
                if torch.isnan(loss):
                    breakpoint()

                accu_loss += loss

                if (
                    id % 32 == 31 or id == len(train_dataloader) - 1
                ):  # BP every 32 sample
                    loss.backward()
                    self.optimizer_diffuser.step()
                    self.optimizer_history.step()
                    self.lr_scheduler_diffuser.step()
                    self.optimizer_diffuser.zero_grad()
                    self.optimizer_history.zero_grad()

                    bsz_len = id % 32 + 1

                    wandb.log(
                        {"train_loss/loss": accu_loss.detach().item() / bsz_len},
                        step=global_step,
                    )
                    wandb.log(
                        {"train_loss/diffuser_lr": self.lr_scheduler_diffuser.get_lr()},
                        step=global_step,
                    )

                    accu_loss = torch.tensor(0).float().to(self.device)

                loss = loss.detach().item()
                if epoch % 10 == 0 and loss < min_loss:
                    min_loss = loss
                    torch.save(self.model.state_dict(), self.config.save_ckpt_path)
                losses.append(loss)

            # Wandb
            if epoch % 10 == 0:
                # TODO: multimodal eval
                # Trainset val Metric
                mrmse = 0
                mcos = 0
                mmag = 0
                mflow = 0
                repeat_time = 1
                for i in range(repeat_time):
                    # metric = self.predict(train_val_dataloader, vis=False)
                    metric = self.predict_pseudo_history(train_val_dataloader)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mflow += metric["flow_loss"]
                wandb.log({"train/rmse": mrmse / repeat_time}, step=global_step)
                wandb.log({"train/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"train/mag": mmag / repeat_time}, step=global_step)
                wandb.log({"train/flow": mflow / repeat_time}, step=global_step)

                # Validation Metric - Always with history
                mrmse = 0
                mcos = 0
                mmag = 0
                mflow = 0
                repeat_time = 1
                for i in range(repeat_time):
                    # metric = self.predict(val_dataloader, vis=False)
                    metric = self.predict_pseudo_history(val_dataloader)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mflow += metric["flow_loss"]
                wandb.log(
                    {"Pseudo-History/rmse": mrmse / repeat_time}, step=global_step
                )
                wandb.log({"Pseudo-History/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"Pseudo-History/mag": mmag / repeat_time}, step=global_step)
                wandb.log(
                    {"Pseudo-History/flow": mflow / repeat_time}, step=global_step
                )

                # Test Metric - Start from fully closed, obtain feedback, and predict until it gets correct
                mrmse = 0
                mcos = 0
                mmag = 0
                # mflow = 0
                mstep = 0
                repeat_time = 1
                for i in range(repeat_time):
                    # metric = self.predict(unseen_dataloader, vis=False)
                    metric = self.predict_real_history(unseen_dataloader, max_step=20)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mstep += metric["step"]
                    # mflow += metric["flow_loss"]
                wandb.log({"Real-History/rmse": mrmse / repeat_time}, step=global_step)
                wandb.log({"Real-History/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"Real-History/mag": mmag / repeat_time}, step=global_step)
                wandb.log({"Real-History/step": mstep / repeat_time}, step=global_step)

        # Visualize loss
        plt.figure()
        plt.plot(losses[::50])

    def predict_real_history(self, val_dataloader, max_step=20):
        self.model.eval()
        self.history_encoder.eval()

        valid_sample_count = 0

        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        # all_flow_loss = 0
        all_steps = 0

        all_cosines = []

        # Eval every dataloader
        with torch.no_grad():
            for id, sample in tqdm.tqdm(enumerate(val_dataloader)):
                valid_sample_count += 1
                rmse, cos_dist, mag_error, step, cosines = self.predict_step(
                    sample, max_step
                )

                all_steps += step
                all_rmse += rmse.item()
                all_cos_dist += cos_dist.item()
                all_mag_error += mag_error.item()

                all_cosines.append(cosines)
                # print(all_rmse)

        return {
            "rmse": all_rmse / valid_sample_count,
            "cos_dist": all_cos_dist / valid_sample_count,
            "mag_error": all_mag_error / valid_sample_count,
            # "flow_loss": all_flow_loss / valid_sample_count,
            "step": all_steps / valid_sample_count,
            "all_cosines": all_cosines,
        }

    def predict_pseudo_history(self, val_dataloader, vis=False):
        self.model.eval()
        self.history_encoder.eval()

        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        all_flow_loss = 0

        with torch.no_grad():
            for id, sample in enumerate(val_dataloader):
                # The history embedding
                history = torch.cat(
                    [
                        sample.trial_points,
                        sample.trial_directions,
                        sample.trial_results.unsqueeze(-1),
                    ],
                    dim=-1,
                ).unsqueeze(0)
                history_embed = self.history_encoder(history.to(self.device))
                sample.history_embed = history_embed.repeat(self.config.sample_size, 1)

                batch_size = sample.pos.shape[0] // 1200
                noisy_input = (
                    torch.randn((batch_size, 3, self.traj_len, 1200))
                    .float()
                    .to(self.device)
                )
                condition = sample.pos
                condition = condition.reshape(-1, 1200, condition.shape[1]).to(
                    self.device
                )

                mask = sample.mask == 1
                mask = mask.reshape(-1, 1200).to(self.device)

                flow_gt = sample.delta.to(self.device)
                flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])

                flow_gt = normalize_trajectory(
                    torch.flatten(flow_gt, start_dim=0, end_dim=1)
                )
                masked_flow_gt = flow_gt.reshape(
                    -1, 1200, flow_gt.shape[1], flow_gt.shape[2]
                )
                masked_flow_gt = masked_flow_gt[mask == 1]

                if vis:
                    animation = FlowNetAnimation()
                    pcd = sample.pos.numpy()

                for t in self.noise_scheduler.timesteps:
                    model_output = self.model(noisy_input, t, context=sample).sample

                    noisy_input = self.noise_scheduler.step(
                        model_output, t, noisy_input
                    ).prev_sample

                    if vis:
                        if t % 5 == 0 or t == 99:
                            flow = normalize_trajectory(
                                torch.flatten(
                                    noisy_input.transpose(1, 3),
                                    start_dim=0,
                                    end_dim=1,
                                )
                            )[:, 0, :]
                            animation.add_trace(
                                torch.as_tensor(pcd),
                                torch.as_tensor(
                                    [pcd[mask.squeeze().detach().cpu().numpy()]]
                                ),
                                torch.as_tensor(
                                    [
                                        flow[mask.detach().cpu().numpy()]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    ]
                                ),
                                "red",
                            )

                flow_prediction = noisy_input.transpose(1, 3)
                flow_prediction = normalize_trajectory(
                    torch.flatten(flow_prediction, start_dim=0, end_dim=1)
                )
                masked_flow_prediction = flow_prediction.reshape(
                    -1, 1200, flow_prediction.shape[1], flow_prediction.shape[2]
                )
                masked_flow_prediction = masked_flow_prediction[mask == 1]
                n_nodes = torch.as_tensor([d.num_nodes for d in sample.to_data_list()]).to(self.device)  # type: ignore
                flow_loss = artflownet_loss(flow_prediction, flow_gt, n_nodes)

                # Compute some metrics on flow-only regions.
                rmse, cos_dist, mag_error = flow_metrics(
                    masked_flow_prediction, masked_flow_gt
                )

                all_rmse += rmse.item()
                all_cos_dist = cos_dist.item()
                all_mag_error = mag_error.item()
                all_flow_loss = flow_loss.item()

        if vis:
            # fig = animation.animate()
            # fig.show()

            return {
                "animation": animation,
                "rmse": all_rmse / len(val_dataloader),
                "cos_dist": all_cos_dist / len(val_dataloader),
                "mag_error": all_mag_error / len(val_dataloader),
                "flow_loss": all_flow_loss / len(val_dataloader),
            }
        else:
            return {
                "rmse": all_rmse / len(val_dataloader),
                "cos_dist": all_cos_dist / len(val_dataloader),
                "mag_error": all_mag_error / len(val_dataloader),
                "flow_loss": all_flow_loss / len(val_dataloader),
            }

    # TODO: change to adapt to history
    @torch.no_grad()
    def predict_step(self, sample, max_step=20):  # For a single sample
        self.model.eval()
        self.history_encoder.eval()

        batch_size = sample.pos.shape[0] // 1200
        assert batch_size == 1, f"batch size should be 1, now is {batch_size}"

        cosines = []

        # The history embedding - start with no history
        history_stack = []
        history = torch.zeros(batch_size, 1, 7).to(self.device)
        history_embed = self.history_encoder(history)
        sample.history_embed = history_embed.repeat(self.config.sample_size, 1)

        for step in range(max_step):
            noisy_input = (
                torch.randn((batch_size, 3, self.traj_len, 1200))
                .float()
                .to(self.device)
            )

            condition = sample.pos
            condition = condition.reshape(-1, 1200, condition.shape[1]).to(self.device)

            for t in self.noise_scheduler.timesteps:
                model_output = self.model(noisy_input, t, context=sample).sample

                noisy_input = self.noise_scheduler.step(
                    model_output, t, noisy_input
                ).prev_sample

            flow_prediction = noisy_input.transpose(
                1, 3
            ).squeeze()  ## TODO: only support batch size = 1, traj length = 1

            # Pseudo trial and feedback
            grasp_point_id = flow_prediction.norm(dim=-1).argmax()
            grasp_direction = flow_prediction[grasp_point_id]
            gt_direction = sample.delta[grasp_point_id, 0].to(self.device)
            pseudo_feedback = torch.sum(grasp_direction * gt_direction)

            # If success
            pseudo_result = torch.cosine_similarity(
                flow_prediction, sample.delta.squeeze(), dim=-1
            ).mean()
            cosines.append(pseudo_result)
            if pseudo_result > 0.7:  # Success
                break
            history_stack.append(
                torch.cat(
                    [grasp_direction, gt_direction, pseudo_feedback.unsqueeze(0)],
                    dim=-1,
                )
            )

            # Update history
            history = torch.stack(history_stack, dim=0).unsqueeze(0).to(self.device)
            history_embed = self.history_encoder(history)
            sample.history_embed = history_embed.repeat(self.config.sample_size, 1)

        # Metric
        new_flow_prediction = normalize_trajectory(flow_prediction.unsqueeze(1))
        new_flow_prediction = new_flow_prediction.reshape(
            -1, 1200, new_flow_prediction.shape[1], new_flow_prediction.shape[2]
        )

        flow_gt = sample.delta.to(self.device)
        flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])

        flow_gt = normalize_trajectory(torch.flatten(flow_gt, start_dim=0, end_dim=1))
        flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])

        n_nodes = torch.as_tensor([d.num_nodes for d in sample.to_data_list()]).to(self.device)  # type: ignore
        # flow_loss = artflownet_loss(flow_prediction, flow_gt, n_nodes)

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(new_flow_prediction, flow_gt)

        return rmse, cos_dist, mag_error, step, cosines


# TODO: change to adapt to history
class TrajHistoryDiffuserSimWrapper(L.LightningModule):
    def __init__(self, diffuser):
        super().__init__()
        self.diffuser = diffuser

    def forward(self, data):
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = data
        data = tgd.Data(
            pos=torch.from_numpy(P_world).float().cuda(),
            # mask=torch.ones(P_world.shape[0]).float(),
        )
        batch = tgd.Batch.from_data_list([data])
        self.eval()
        with torch.no_grad():
            flow = self.diffuser.predict_step(batch)
        return flow.squeeze().cpu()


if __name__ == "__main__":
    config = TrainingConfig()

    wandb.init(
        entity="r-pad",
        # entity="leisure-thu-cv",
        project="open_anything_diffusion",
        # group="diffusion-PN++",
        # job_type="overfit_trajectory",
        group="door_condition_diffusion",
        job_type="train_condition_diffuser(pseudo feedback)",
    )

    # Create dataset
    toy_dataset = {
        "id": "door-full-new",
        "train-train": [
            "8877",
            "8893",
            "8897",
            "8903",
            "8919",
            "8930",
            "8961",
            "8997",
            "9016",
            "9032",
            "9035",
            "9041",
            "9065",
            "9070",
            "9107",
            "9117",
            "9127",
            "9128",
            "9148",
            "9164",
            "9168",
            "9277",
            "9280",
            "9281",
            "9288",
            "9386",
            "9388",
            "9410",
        ],
        "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
        "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
    }

    # Trajectory module
    trajectory_module = FlowTrajectoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=1,
        num_workers=30,
        n_proc=2,
        seed=42,
        trajectory_len=1,  # Only used when training trajectory model
        toy_dataset=toy_dataset,
    )

    # History Trajectory module
    datamodule = FlowHistoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=1,
        num_workers=30,
        n_proc=2,
        trajectory_datasets=trajectory_module,
        seed=42,
        trajectory_len=1,  # Only used when inference trajectory model
        toy_dataset=toy_dataset,
    )

    train_dataloader = datamodule.train_dataloader()
    train_val_dataloader = datamodule.train_val_dataloader(bsz=1)
    val_dataloader = datamodule.val_dataloader(bsz=1)
    unseen_dataloader = datamodule.unseen_dataloader(bsz=1)

    diffuser = HistoryTrajDiffuser(config, train_batch_num=len(train_dataloader))

    # Train
    # diffuser.load_model('/home/yishu/open_anything_diffusion/src/open_anything_diffusion/models/diffusion/door_diffusion_multimodal_ckpt.pth')
    diffuser.train(
        train_dataloader, train_val_dataloader, val_dataloader, unseen_dataloader
    )

    # # Overfit
    # datamodule = FlowTrajectoryDataModule(
    #     root="/home/yishu/datasets/partnet-mobility",
    #     batch_size=1,
    #     num_workers=30,
    #     n_proc=2,
    #     seed=42,
    #     trajectory_len=config.traj_len,  # Only used when training trajectory model
    # )

    # train_dataloader = datamodule.train_dataloader()
    # val_dataloader = datamodule.train_val_dataloader()

    # # # Overfit
    # samples = list(enumerate(train_dataloader))
    # # breakpoint()
    # sample = samples[0][1]
    # diffuser.train([sample], [sample])

    wandb.finish()

    # diffuser.load_model('/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-08-31/01-21-42/checkpoints/epoch=199-step=157200.ckpt')

    # ##  Overfit sample prediction
    # metric = diffuser.predict(sample, vis=True)
    # print(f"dist:{metric['mean_dist']} cos:{metric['cos_sim']}")

    # # Permutated sample prediction
    # indices = torch.randperm(1200)
    # sample.pos = sample.pos[indices]
    # sample.delta = sample.delta[indices]
    # sample.mask = sample.mask[indices]
    # metric = diffuser.predict(sample, vis=True)
    # print(f"dist:{metric['mean_dist']} cos:{metric['cos_sim']}")
