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

from open_anything_diffusion.metrics.trajectory import (
    artflownet_loss,
    flow_metrics,
    normalize_trajectory,
)
from open_anything_diffusion.models.diffusion.model import PNDiffuser

# import open_anything_diffusion.models.diffusion.module as pnp


@dataclass
class TrainingConfig:
    device = "cuda"

    image_size = 128  # the generated image resolution
    batch_size = 32
    # train_batch_size = 16
    # eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 1000
    # num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 10
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    train_sample_number = 1

    traj_len = 1
    # Diffuser params
    num_train_timesteps = 10
    seed = 0
    sample_size = [1, 1200]
    in_channels = 3
    out_channels = 3
    cross_attention_dim = 3
    block_out_channels = [128, 256, 512, 512]
    attention_head_dim = 3

    # ckpt params
    read_ckpt_path = "./diffusion_synthetic_ckpt.pth"
    save_ckpt_path = "./diffusion_synthetic_ckpt.pth"


class TrajDiffuser:
    def __init__(self, config, train_batch_num):
        self.config = config
        self.traj_len = config.traj_len
        self.device = config.device

        # self.model = UNet2DConditionModel(
        #     sample_size=config.sample_size,
        #     in_channels=config.in_channels,
        #     out_channels=config.out_channels,
        #     cross_attention_dim=config.cross_attention_dim,
        #     block_out_channels=config.block_out_channels,
        #     attention_head_dim=config.attention_head_dim,
        # ).to(config.device)

        # self.model = DGCNN(
        #          in_channels=3 * config.traj_len,
        #          sample_size=1200,
        #          time_embed_dim=64,
        #          emb_dims=3).to(config.device)

        self.model = PNDiffuser(
            in_channels=3 * config.traj_len,
            # sample_size=1200,
            traj_len=config.traj_len,
            time_embed_dim=64,
            # emb_dims=3
        ).to(config.device)

        # self.model = pnp.PN2Dense(
        #     in_channels=3 * (self.traj_len + 1),   # noise concatenated with condition signal
        #     out_channels=3 * self.traj_len,
        #     p=pnp.PN2DenseParams(),
        # )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(train_batch_num * config.num_epochs),
            # num_training_steps=((config.train_sample_number // config.batch_size) * config.num_epochs),
            # num_training_steps=(config.num_epochs),
        )

    def load_model(self, ckpt_path="./diffusion_best_ckpt.pth"):
        self.model.load_state_dict(torch.load(ckpt_path))

    def train(
        self, train_dataloader, train_val_dataloader, val_dataloader, unseen_dataloader
    ):  # TODO:currently only support overfit
        ## Train loop
        losses = []
        min_loss = 100
        global_step = 0
        # clean_flow = torch.tensor(sample.delta.transpose(0, 2).unsqueeze(0)).to(
        #     self.device
        # )
        # condition = torch.tensor(sample.pos.unsqueeze(0)).to(self.device)
        for epoch in range(config.num_epochs):
            print(f"Epoch: {epoch}")
            self.model.train()
            for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
                global_step += 1

                clean_flow = batch.delta
                clean_flow = clean_flow.reshape(
                    -1, 1200, clean_flow.shape[1], clean_flow.shape[2]
                ).to(self.device)
                condition = batch.pos
                condition = condition.reshape(-1, 1200, condition.shape[1]).to(
                    self.device
                )
                # breakpoint()

                # Random permutation
                # perm = torch.randperm(1200).to(self.device)
                # clean_flow = clean_flow[:, perm]
                # condition = condition[:, perm]

                clean_flow = clean_flow.transpose(1, 3)

                # breakpoint()

                # Sample noise to add to the images
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
                # noise_pred = self.model(
                #     noisy_images,
                #     encoder_hidden_states=condition,
                #     timestep=timesteps,
                #     return_dict=False,
                # )[0]
                # model_input.pos = condition
                noise_pred = self.model(
                    noisy_images, timesteps, context=batch, return_dict=False
                )[0]

                # print("train: ", noisy_images[:, :, :, :2], noise_pred[:, :, :, :2])
                # print("pred: ", noise_pred[:2])
                # print("noisy_images: ", noisy_images[:2])
                # print("noise: ", noise[:2])
                # breakpoint()
                loss = F.mse_loss(noise_pred, noise)

                if global_step % 1 == 0:
                    wandb.log({"train_loss/loss": loss}, step=global_step)
                    wandb.log(
                        {"train_loss/lr": self.lr_scheduler.get_lr()}, step=global_step
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                loss = loss.detach().item()
                # if epoch % 10 == 0 and loss < min_loss:
                #     min_loss = loss
                #     torch.save(self.model.state_dict(), self.config.save_ckpt_path)
                losses.append(loss)
                # print("loss", loss.detach().item(), "lr", lr_scheduler.get_last_lr()[0], "step", epoch)

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
                    metric = self.predict_wta(train_val_dataloader, trial_times=20)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mflow += metric["flow_loss"]
                wandb.log({"train/rmse": mrmse / repeat_time}, step=global_step)
                wandb.log({"train/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"train/mag": mmag / repeat_time}, step=global_step)
                wandb.log({"train/flow": mflow / repeat_time}, step=global_step)
                wandb.log({"train/multimodal": metric["multimodal"]}, step=global_step)

                # Validation Metric
                mrmse = 0
                mcos = 0
                mmag = 0
                mflow = 0
                repeat_time = 1
                for i in range(repeat_time):
                    # metric = self.predict(val_dataloader, vis=False)
                    metric = self.predict_wta(val_dataloader, trial_times=20)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mflow += metric["flow_loss"]
                wandb.log({"val/rmse": mrmse / repeat_time}, step=global_step)
                wandb.log({"val/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"val/mag": mmag / repeat_time}, step=global_step)
                wandb.log({"val/flow": mflow / repeat_time}, step=global_step)
                wandb.log({"val/multimodal": metric["multimodal"]}, step=global_step)

                if epoch % 10 == 0 and mflow < min_loss:
                    min_loss = mflow
                    torch.save(self.model.state_dict(), self.config.save_ckpt_path)

                # Test Metric
                mrmse = 0
                mcos = 0
                mmag = 0
                mflow = 0
                repeat_time = 1
                for i in range(repeat_time):
                    # metric = self.predict(unseen_dataloader, vis=False)
                    metric = self.predict_wta(unseen_dataloader, trial_times=20)
                    mrmse += metric["rmse"]
                    mcos += metric["cos_dist"]
                    mmag += metric["mag_error"]
                    mflow += metric["flow_loss"]
                wandb.log({"unseen/rmse": mrmse / repeat_time}, step=global_step)
                wandb.log({"unseen/cos": mcos / repeat_time}, step=global_step)
                wandb.log({"unseen/mag": mmag / repeat_time}, step=global_step)
                wandb.log({"unseen/flow": mflow / repeat_time}, step=global_step)

        # Visualize loss
        plt.figure()
        plt.plot(losses[::50])

    def predict_wta(self, val_dataloader, trial_times=20):
        self.model.eval()

        valid_sample_count = 0

        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        all_flow_loss = 0
        all_multimodal = 0

        all_directions = []  # dataloader * trial_times

        # Eval every dataloader
        with torch.no_grad():
            for id, orig_sample in tqdm.tqdm(enumerate(val_dataloader)):
                # breakpoint()
                batch_size = orig_sample.pos.shape[0] // 1200
                assert batch_size == 1, f"batch size should be 1, now is {batch_size}"

                # batch every sample into bsz of trial_times
                data_list = orig_sample.to_data_list() * trial_times
                sample = tgd.Batch.from_data_list(data_list)
                batch_size = trial_times

                # breakpoint()

                noisy_input = (
                    torch.randn((batch_size, 3, self.traj_len, 1200))
                    .float()
                    .to(self.device)
                )

                condition = sample.pos
                # breakpoint()
                condition = condition.reshape(-1, 1200, condition.shape[1]).to(
                    self.device
                )

                mask = sample.mask == 1
                if torch.sum(mask) == 0:  # Skip those unmovable samples
                    continue
                valid_sample_count += 1

                mask = mask.reshape(-1, 1200).to(self.device)
                # sample.mask = mask.repeat(batch_size, 1)

                flow_gt = sample.delta.to(self.device)
                flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])
                # flow_gt = flow_gt.repeat(batch_size, 1, 1, 1)
                # breakpoint()

                # # Random permutation
                # perm = torch.randperm(1200).to(self.device)
                # mask = mask[:, perm]
                # flow_gt = flow_gt[:, perm]
                # condition = condition[:, perm]

                flow_gt = normalize_trajectory(
                    torch.flatten(flow_gt, start_dim=0, end_dim=1)
                )
                masked_flow_gt = flow_gt.reshape(
                    -1, 1200, flow_gt.shape[1], flow_gt.shape[2]
                )
                masked_flow_gt = masked_flow_gt[mask == 1]

                for t in self.noise_scheduler.timesteps:
                    # model_output = self.model(
                    #     noisy_input, encoder_hidden_states=condition, timestep=t
                    # ).sample
                    # breakpoint()
                    model_output = self.model(noisy_input, t, context=sample).sample

                    noisy_input = self.noise_scheduler.step(
                        model_output, t, noisy_input
                    ).prev_sample
                #     print(t, model_output.transpose(1, 3)[0, :2, 0, :], noisy_input.transpose(1, 3)[0, :2, 0, :])

                # print("-------------------")
                # print("-------------------")
                # print("-------------------")
                # print("-------------------")

                flow_prediction = noisy_input.transpose(1, 3)
                flow_prediction = normalize_trajectory(
                    torch.flatten(flow_prediction, start_dim=0, end_dim=1)
                )

                masked_flow_prediction = flow_prediction.reshape(
                    -1, 1200, flow_prediction.shape[1], flow_prediction.shape[2]
                )
                masked_flow_prediction = masked_flow_prediction[mask == 1]
                n_nodes = torch.as_tensor([d.num_nodes for d in sample.to_data_list()]).to(self.device)  # type: ignore

                flow_loss = artflownet_loss(
                    flow_prediction, flow_gt, n_nodes, reduce=False
                )

                # Compute some metrics on flow-only regions.
                rmse, cos_dist, mag_error = flow_metrics(
                    masked_flow_prediction, masked_flow_gt, reduce=False
                )

                # Aggregate the results
                # Choose the one with smallest flow loss
                flow_loss = flow_loss.reshape(batch_size, -1).mean(-1)
                rmse = rmse.reshape(batch_size, -1).mean(-1)
                cos_dist = cos_dist.reshape(batch_size, -1).mean(-1)
                all_directions += list(cos_dist)

                mag_error = mag_error.reshape(batch_size, -1).mean(-1)

                # breakpoint()
                chosen_id = torch.min(flow_loss, 0)[1]  # index
                chosen_direction = cos_dist[chosen_id]
                if chosen_direction > 0:
                    multimodal = torch.sum((cos_dist + 0.3) < 0) != 0  # < 0.3
                else:
                    multimodal = torch.sum((cos_dist - 0.3) > 0) != 0  # > 0.3

                # print(multimodal, rmse[chosen_id], cos_dist[chosen_id], mag_error[chosen_id], flow_loss[chosen_id])
                all_multimodal += multimodal.item()
                all_rmse += rmse[chosen_id].item()
                all_cos_dist += cos_dist[chosen_id].item()
                all_mag_error += mag_error[chosen_id].item()
                all_flow_loss += flow_loss[chosen_id].item()

                # print(all_rmse)

        return {
            "rmse": all_rmse / valid_sample_count,
            "cos_dist": all_cos_dist / valid_sample_count,
            "mag_error": all_mag_error / valid_sample_count,
            "flow_loss": all_flow_loss / valid_sample_count,
            "multimodal": all_multimodal / valid_sample_count,
            "all_directions": all_directions,
        }

    def predict(self, val_dataloader, vis=False):
        self.model.eval()

        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        all_flow_loss = 0

        with torch.no_grad():
            for id, sample in enumerate(val_dataloader):
                # breakpoint()
                batch_size = sample.pos.shape[0] // 1200
                noisy_input = (
                    torch.randn((batch_size, 3, self.traj_len, 1200))
                    .float()
                    .to(self.device)
                )
                # condition = condition
                # breakpoint()
                condition = sample.pos
                condition = condition.reshape(-1, 1200, condition.shape[1]).to(
                    self.device
                )

                mask = sample.mask == 1
                mask = mask.reshape(-1, 1200).to(self.device)

                flow_gt = sample.delta.to(self.device)
                flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])

                # breakpoint()

                # # Random permutation
                # perm = torch.randperm(1200).to(self.device)
                # mask = mask[:, perm]
                # flow_gt = flow_gt[:, perm]
                # condition = condition[:, perm]

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
                    # model_output = self.model(
                    #     noisy_input, encoder_hidden_states=condition, timestep=t
                    # ).sample
                    # breakpoint()
                    model_output = self.model(noisy_input, t, context=sample).sample

                    noisy_input = self.noise_scheduler.step(
                        model_output, t, noisy_input
                    ).prev_sample

                    if vis:
                        # print(noisy_input.shape)
                        # print(torch.nn.functional.normalize(noisy_input, p=2, dim=1)
                        #     .squeeze().permute(1, 0).shape)
                        # print(torch.flatten(noisy_input.transpose(1, 3), start_dim=0, end_dim=1).shape)
                        if t % 5 == 0 or t == 99:
                            flow = (
                                # torch.nn.functional.normalize(noisy_input, p=2, dim=1)
                                # .squeeze()
                                # .permute(1, 0)
                                normalize_trajectory(
                                    torch.flatten(
                                        noisy_input.transpose(1, 3),
                                        start_dim=0,
                                        end_dim=1,
                                    )
                                )[:, 0, :]
                            )
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
                # flow_prediction = torch.nn.functional.normalize(
                #     flow_prediction, p=2, dim=-1
                # )
                # largest_mag: float = torch.linalg.norm(
                #     flow_prediction, ord=2, dim=-1
                # ).max()
                # flow_prediction = flow_prediction / (largest_mag + 1e-6)
                # flow_prediction = flow_prediction.permute(1, 2, 0)
                masked_flow_prediction = masked_flow_prediction[mask == 1]
                # breakpoint()

                # mean_dist = (masked_flow_prediction - flow_gt).norm(p=2, dim=-1).mean()
                # cos_sim = torch.cosine_similarity(
                #     masked_flow_prediction, flow_gt, dim=-1
                # ).mean()
                n_nodes = torch.as_tensor([d.num_nodes for d in sample.to_data_list()]).to(self.device)  # type: ignore
                # breakpoint()
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

    def predict_step(self, sample):  # For a single sample
        batch_size = sample.pos.shape[0] // 1200
        noisy_input = (
            torch.randn((batch_size, 3, self.traj_len, 1200)).float().to(self.device)
        )

        condition = sample.pos
        condition = condition.reshape(-1, 1200, condition.shape[1]).to(self.device)

        for t in self.noise_scheduler.timesteps:
            model_output = self.model(noisy_input, t, context=sample).sample

            noisy_input = self.noise_scheduler.step(
                model_output, t, noisy_input
            ).prev_sample

        flow_prediction = noisy_input.transpose(1, 3)

        # # Metric

        # new_flow_prediction = normalize_trajectory(
        #         torch.flatten(flow_prediction, start_dim=0, end_dim=1)
        #     )
        # new_flow_prediction = new_flow_prediction.reshape(
        #     -1, 1200, new_flow_prediction.shape[1], new_flow_prediction.shape[2]
        # )

        # flow_gt = sample.delta.to(self.device)
        # flow_gt = flow_gt.reshape(-1, 1200, flow_gt.shape[1], flow_gt.shape[2])

        # flow_gt = normalize_trajectory(
        #     torch.flatten(flow_gt, start_dim=0, end_dim=1)
        # )
        # flow_gt = flow_gt.reshape(
        #     -1, 1200, flow_gt.shape[1], flow_gt.shape[2]
        # )

        # n_nodes = torch.as_tensor([d.num_nodes for d in sample.to_data_list()]).to(self.device)  # type: ignore
        # # breakpoint()
        # flow_loss = artflownet_loss(flow_prediction, flow_gt, n_nodes)

        # # Compute some metrics on flow-only regions.
        # rmse, cos_dist, mag_error = flow_metrics(
        #     new_flow_prediction, flow_gt
        # )

        # print("flow_loss: ", flow_loss)
        # print("rmse, cos_dist, mag_error: ", rmse, cos_dist, mag_error)
        return flow_prediction


class TrajDiffuserSimWrapper(L.LightningModule):
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
        group="synthetic",
        job_type="train_diffuser_wta",
    )

    from open_anything_diffusion.datasets.synthetic_dataset import SyntheticDataModule

    datamodule = SyntheticDataModule(batch_size=1, seed=42)

    train_dataloader = datamodule.train_dataloader()
    train_val_dataloader = datamodule.train_val_dataloader(bsz=1)
    val_dataloader = datamodule.val_dataloader(bsz=1)
    unseen_dataloader = datamodule.unseen_dataloader(bsz=1)

    diffuser = TrajDiffuser(config, train_batch_num=len(train_dataloader))

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
