# Find multimodal cases in diffusion models

import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation
from hydra import compose, initialize

from open_anything_diffusion.metrics.trajectory import (
    flow_metrics,
    normalize_trajectory,
)
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffusionModule,
)

initialize(config_path="../configs", version_base="1.3")
cfg = compose(config_name="train")


ckpt_path = "/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-08-31/16-13-10/checkpoints/epoch=394-step=310470-val_loss=0.00-weights-only.ckpt"
network = pnp.PN2Dense(
    in_channels=67,
    out_channels=3,
    p=pnp.PN2DenseParams(),
)

model = FlowTrajectoryDiffusionModule(network, cfg.training, cfg.model)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt["state_dict"])
model = model.cuda()


import torch_geometric.loader as tgl

from open_anything_diffusion.datasets.flow_trajectory_dataset_pyg import (
    FlowTrajectoryPyGDataset,
)

datamodule = FlowTrajectoryPyGDataset(
    root="/home/yishu/datasets/partnet-mobility/raw",
    split="umpnet-train-test",
    randomize_joints=True,
    randomize_camera=True,
    # batch_size=1,
    # num_workers=30,
    # n_proc=2,
    seed=42,
    trajectory_len=cfg.training.trajectory_len,  # Only used when training trajectory model
)
val_dataloader = tgl.DataLoader(datamodule, 1, shuffle=False, num_workers=0)

samples = list(enumerate(val_dataloader))


@torch.no_grad()
def diffuse_visual(batch, model):  # 1 sample batch
    model.eval()

    animation = FlowNetAnimation()
    pcd = batch.pos.cpu().numpy()
    mask = batch.mask.cpu().long().numpy()

    fix_noise = torch.randn_like(batch.delta, device="cuda")

    bs = batch.delta.shape[0] // 1200
    # batch.traj_noise = torch.randn_like(batch.delta, device="cuda")
    batch.traj_noise = fix_noise
    # batch.traj_noise = normalize_trajectory(batch.traj_noise)
    # breakpoint()

    # import time
    # batch_time = 0
    # model_time = 0
    # noise_scheduler_time = 0
    # self.noise_scheduler_inference.set_timesteps(self.num_inference_timesteps)
    # print(self.noise_scheduler_inference.timesteps)
    # for t in self.noise_scheduler_inference.timesteps:
    for t in model.noise_scheduler.timesteps:
        # tm = time.time()
        batch.timesteps = torch.zeros(bs, device=model.device) + t  # Uniform t steps
        batch.timesteps = batch.timesteps.long()
        # batch_time += time.time() - tm

        # tm = time.time()
        model_output = model(batch)  # bs * 1200, traj_len * 3
        model_output = model_output.reshape(
            model_output.shape[0], -1, 3
        )  # bs * 1200, traj_len, 3

        batch.traj_noise = model.noise_scheduler.step(
            # batch.traj_noise = self.noise_scheduler_inference.step(
            model_output.reshape(
                -1, model.sample_size, model_output.shape[1], model_output.shape[2]
            ),
            t,
            batch.traj_noise.reshape(
                -1, model.sample_size, model_output.shape[1], model_output.shape[2]
            ),
        ).prev_sample
        batch.traj_noise = torch.flatten(batch.traj_noise, start_dim=0, end_dim=1)

        # print(batch.traj_noise)
        if t % 50 == 0:
            flow = batch.traj_noise.squeeze().cpu().numpy()
            # print(flow[mask])
            # segmented_flow = np.zeros_like(flow, dtype=np.float32)
            # segmented_flow[mask] = flow[mask]
            # print("seg", segmented_flow, "flow", flow)
            animation.add_trace(
                torch.as_tensor(pcd),
                # torch.as_tensor([pcd[mask]]),
                # torch.as_tensor([flow[mask].detach().cpu().numpy()]),
                torch.as_tensor([pcd]),
                torch.as_tensor([flow]),
                "red",
            )

    f_pred = batch.traj_noise
    f_pred = normalize_trajectory(f_pred)
    # largest_mag: float = torch.linalg.norm(
    #     f_pred, ord=2, dim=-1
    # ).max()
    # f_pred = f_pred / (largest_mag + 1e-6)

    # Compute the loss.
    n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to("cuda")  # type: ignore
    f_ix = batch.mask.bool()
    f_target = batch.delta
    f_target = normalize_trajectory(f_target)

    f_target = f_target.float()
    # loss = artflownet_loss(f_pred, f_target, n_nodes)

    # Compute some metrics on flow-only regions.
    rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], batch.delta[f_ix])

    return cos_dist, animation


# repeat_times = 10
# for sample in tqdm.tqdm(samples):
#     sample_id = sample[0]
#     sample = sample[1]
#     batch = sample.cuda()
#     has_correct = False
#     correct_dist = 0
#     has_incorrect = False
#     incorrect_dist = 10
#     for _ in range(repeat_times):
#         cos_dist, animation = diffuse_visual(batch, model)
#         if cos_dist > 0.5:
#             correct_dist = max(correct_dist, cos_dist)
#             has_correct = True
#             correct_animation = animation
#         elif cos_dist < -0.5:
#             incorrect_dist = min(incorrect_dist, cos_dist)
#             has_incorrect = True
#             incorrect_animation = animation
#     if has_correct and has_incorrect:
#         print(sample_id, correct_dist, incorrect_dist)

repeat_times = 3
mean_cos_dist = 0
count = 0
for sample in tqdm.tqdm(samples):
    sample_id = sample[0]
    sample = sample[1]
    batch = sample.cuda()
    best_cos_dist = -1
    count += 1
    for _ in range(repeat_times):
        cos_dist, animation = diffuse_visual(batch, model)
        best_cos_dist = max(best_cos_dist, cos_dist)

    print(best_cos_dist)
    mean_cos_dist += best_cos_dist

mean_cos_dist /= count
print(mean_cos_dist)
