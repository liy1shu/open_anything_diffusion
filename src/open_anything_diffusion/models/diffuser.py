## Diffusion model
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
import wandb
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule


@dataclass
class TrainingConfig:
    device = "cuda"

    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 30000
    # num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    train_sample_number = 1

    # Diffuser params
    num_train_timesteps = 1000
    seed = 0
    sample_size = [1, 1200]
    in_channels = 3
    out_channels = 3
    cross_attention_dim = 3
    block_out_channels = [128, 256, 512, 512]
    attention_head_dim = 3

    # ckpt params
    read_ckpt_path = "./diffusion_best_ckpt.pth"
    save_ckpt_path = "./diffusion_try_best_ckpt.pth"


class TrajDiffuser:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.model = UNet2DConditionModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            cross_attention_dim=config.cross_attention_dim,
            block_out_channels=config.block_out_channels,
            attention_head_dim=config.attention_head_dim,
        ).to(config.device)

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

    def train(self, sample):  # TODO:currently only support overfit
        ## Train loop
        losses = []
        min_loss = 10

        clean_flow = torch.tensor(sample.delta.transpose(0, 2).unsqueeze(0)).to(
            self.device
        )
        condition = torch.tensor(sample.pos.unsqueeze(0)).to(self.device)

        for epoch in tqdm.tqdm(range(config.num_epochs)):
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
            noisy_images = self.noise_scheduler.add_noise(clean_flow, noise, timesteps)

            # Predict the noise residual
            noise_pred = self.model(
                noisy_images,
                encoder_hidden_states=condition,
                timestep=timesteps,
                return_dict=False,
            )[0]
            loss = F.mse_loss(noise_pred, noise)

            # Wandb
            if epoch % 300 == 0:
                wandb.log({"train/loss": loss}, step=epoch)

                # Validation Metric
                mdist = 0
                msim = 0
                for i in range(5):
                    metric = self.predict(sample, vis=False)
                    mdist += metric["mean_dist"]
                    msim += metric["cos_sim"]
                wandb.log({"dist": mdist / 5}, step=epoch)
                wandb.log({"cos": msim / 5}, step=epoch)

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

    def predict(self, sample, vis=False):
        noisy_input = torch.randn((1, 3, 1, 1200)).float().to(self.device)
        # condition = condition
        condition = sample.pos.unsqueeze(0).to(self.device)  # 1 * 1200 * 3
        mask = sample.mask == 1

        if vis:
            animation = FlowNetAnimation()
            pcd = sample.pos.numpy()

        with torch.no_grad():
            for t in tqdm.tqdm(self.noise_scheduler.timesteps):
                model_output = self.model(
                    noisy_input, encoder_hidden_states=condition, timestep=t
                ).sample
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

        flow_prediction = noisy_input.squeeze().transpose(1, 0)
        flow_prediction = torch.nn.functional.normalize(flow_prediction, p=2, dim=1)
        masked_flow_prediction = flow_prediction[mask == 1]

        flow_gt = sample.delta.squeeze()[mask == 1].to(self.device)

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
        project="open_anything_diffusion",
        group="diffusion-overfit",
        job_type="train",
    )

    datamodule = FlowTrajectoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=64,
        num_workers=30,
        n_proc=2,
        seed=42,
        trajectory_len=1,  # Only used when training trajectory model
    )

    train_dataloader = datamodule.train_val_dataloader()

    samples = list(enumerate(train_dataloader))
    sample = samples[0][1].get_example(0)

    ## Train
    diffuser.train(sample)

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
