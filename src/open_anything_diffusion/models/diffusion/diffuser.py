## Diffusion model
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import rpad.pyg.nets.pointnet2 as pnp
import tqdm
import wandb
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule
from open_anything_diffusion.models.diffusion.model import PNDiffuser

# import open_anything_diffusion.models.diffusion.module as pnp


@dataclass
class TrainingConfig:
    device = "cuda"

    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100000
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
    sample_size = [1, 1200]
    in_channels = 3
    out_channels = 3
    cross_attention_dim = 3
    block_out_channels = [128, 256, 512, 512]
    attention_head_dim = 3

    # ckpt params
    read_ckpt_path = "./diffusion_best_ckpt.pth"
    save_ckpt_path = "./diffusion_overfit_best_5_ckpt.pth"


class TrajDiffuser:
    def __init__(self, config):
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
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(config.train_sample_number * config.num_epochs),
        )

    def load_model(self, ckpt_path="./diffusion_best_ckpt.pth"):
        self.model.load_state_dict(torch.load(ckpt_path))

    def train(
        self, train_dataloader, val_dataloader
    ):  # TODO:currently only support overfit
        ## Train loop
        losses = []
        min_loss = 10
        global_step = 0
        # clean_flow = torch.tensor(sample.delta.transpose(0, 2).unsqueeze(0)).to(
        #     self.device
        # )
        # condition = torch.tensor(sample.pos.unsqueeze(0)).to(self.device)

        for epoch in range(config.num_epochs):
            print(f"Epoch: {epoch}")
            for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
                # Wandb
                if global_step % 200 == 0:
                    # Validation Metric
                    mdist = 0
                    msim = 0
                    repeat_time = 5
                    for i in range(repeat_time):
                        metric = self.predict(val_dataloader, vis=False)
                        mdist += metric["mean_dist"]
                        msim += metric["cos_sim"]
                    wandb.log({"dist": mdist / repeat_time}, step=global_step)
                    wandb.log({"cos": msim / repeat_time}, step=global_step)

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
                loss = F.mse_loss(noise_pred, noise)

                if global_step % 100 == 0:
                    wandb.log({"train/loss": loss}, step=global_step)

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                loss = loss.detach().item()
                if epoch % 100 == 0 and loss < min_loss:
                    min_loss = loss
                    torch.save(self.model.state_dict(), self.config.save_ckpt_path)
                losses.append(loss)
                # print("loss", loss.detach().item(), "lr", lr_scheduler.get_last_lr()[0], "step", epoch)

        # Visualize loss
        plt.figure()
        plt.plot(losses[::50])

    def predict(self, val_dataloader, vis=False):
        with torch.no_grad():
            for id, sample in tqdm.tqdm(enumerate(val_dataloader)):
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

                flow_gt = flow_gt[mask == 1]

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
                        if t % 20 == 0 or t == 999:
                            flow = (
                                torch.nn.functional.normalize(noisy_input, p=2, dim=1)
                                .squeeze()
                                .permute(1, 0)
                            )
                            animation.add_trace(
                                torch.as_tensor(pcd),
                                torch.as_tensor([pcd[mask]]),
                                torch.as_tensor([flow[mask].detach().cpu().numpy()]),
                                "red",
                            )

                flow_prediction = noisy_input.transpose(1, 3)
                flow_prediction = torch.nn.functional.normalize(
                    flow_prediction, p=2, dim=-1
                )
                # flow_prediction = flow_prediction.permute(1, 2, 0)
                masked_flow_prediction = flow_prediction[mask == 1]

                # breakpoint()

                mean_dist = (masked_flow_prediction - flow_gt).norm(p=2, dim=-1).mean()
                cos_sim = torch.cosine_similarity(
                    masked_flow_prediction, flow_gt, dim=-1
                ).mean()

                if vis:
                    fig = animation.animate()
                    fig.show()

        return {"mean_dist": mean_dist, "cos_sim": cos_sim}


if __name__ == "__main__":
    config = TrainingConfig()
    diffuser = TrajDiffuser(config)

    wandb.init(
        entity="r-pad",
        # entity="leisure-thu-cv",
        project="open_anything_diffusion",
        group="diffusion-PN++",
        job_type="train",
    )

    # datamodule = FlowTrajectoryDataModule(
    #     root="/home/yishu/datasets/partnet-mobility",
    #     batch_size=16,
    #     num_workers=30,
    #     n_proc=2,
    #     seed=42,
    #     trajectory_len=config.traj_len,  # Only used when training trajectory model
    # )

    # Overfit
    datamodule = FlowTrajectoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=1,
        num_workers=30,
        n_proc=2,
        seed=42,
        trajectory_len=config.traj_len,  # Only used when training trajectory model
    )

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.train_val_dataloader()

    # # Overfit
    samples = list(enumerate(train_dataloader))
    # breakpoint()
    sample = samples[0][1]
    diffuser.train([sample], [sample])

    # # Train
    # diffuser.train(train_dataloader, val_dataloader)

    wandb.finish()

    # diffuser.load_model('/home/yishu/diffusion_best_ckpt.pth')

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
