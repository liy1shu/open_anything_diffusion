import numpy as np
import torch
from hydra import compose, initialize

initialize(config_path="../../configs", version_base="1.3")
cfg = compose(config_name="eval_sim")

import copy
import os

import pybullet as p

# Model 0 - FlowBot
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
from rpad.partnet_mobility_utils.data import PMObject

from open_anything_diffusion.metrics.trajectory import flow_metrics
from open_anything_diffusion.models.flow_diffuser_hispndit import (
    FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
)
from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
from open_anything_diffusion.models.modules.dit_models import PN2HisDiT
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder
from open_anything_diffusion.simulations.suction import GTFlowModel, PMSuctionSim

network = pnp.PN2Dense(
    in_channels=0,
    out_channels=3 * cfg.inference.trajectory_len,
    p=pnp.PN2DenseParams(),
)
ckpt = torch.load(
    "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-03-18/12-13-31/checkpoints/epoch=78-step=3476-val_loss=0.00-weights-only.ckpt"
)
# ckpt = torch.load('/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-05-26/02-37-08/checkpoints/epoch=98-step=109395-val_loss=0.00-weights-only.ckpt')
network.load_state_dict(
    {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
)
flowbot = FlowSimulationInferenceModule(network, mask_input_channel=False)
flowbot.eval()

# Model 1 - DiT
# network = DiT(
#     in_channels=3 + 3,
#     depth=5,
#     hidden_size=128,
#     num_heads=4,
#     learn_sigma=True,
# ).cuda()
# model = FlowTrajectoryDiffuserSimulationModule_DiT(
#     network, inference_cfg=cfg.inference, model_cfg=cfg.model
# ).cuda()
# model.eval()

# ckpt = torch.load("/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt",)
# network.load_state_dict(
#     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
# )

# # Model - 2: PN2DiT
# network = PN2DiT(
#     in_channels=3,
#     depth=5,
#     hidden_size=128,
#     patch_size=1,
#     num_heads=4,
#     n_points=1200,
# ).cuda()
# ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_pndit/2024-04-23/05-01-44/checkpoints/epoch=469-step=1038700-val_loss=0.00-weights-only.ckpt"
# model = FlowTrajectoryDiffuserSimulationModule_PNDiT(
#     network, inference_cfg=cfg.inference, model_cfg=cfg.model
# ).cuda()
# model.load_from_ckpt(ckpt_file)
# model.eval()

# Model - 3: HisPNDiT
network = {
    "DiT": PN2HisDiT(
        history_embed_dim=128,
        in_channels=3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        # depth=8,
        # hidden_size=256,
        # num_heads=4,
        learn_sigma=True,
    ).cuda(),
    "History": HistoryEncoder(
        history_dim=128,
        history_len=1,
        batch_norm=True,
        transformer=False,
        repeat_dim=False,
    ).cuda(),
}
ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hispndit/2024-05-25/02-00-54/checkpoints/epoch=399-step=331600-val_loss=0.00-weights-only.ckpt"
model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
    network, inference_cfg=cfg.inference, model_cfg=cfg.model
).cuda()
model.load_from_ckpt(ckpt_file)
model.eval()


# door_ids = ["8877", "8877", "9065", "9410", "8867", "9388", "8893"]
# joint_ids = [0, 1, 0, 0, 0, 0, 1]
door_ids = ["8867"]
joint_ids = [0]
# door_ids = ["8877"]
# joint_ids = [1]
pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")

for obj_id, joint_id in tqdm.tqdm(zip(door_ids, joint_ids)):
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )
    available_joints = [joint.name for joint in available_joints]

    env = PMSuctionSim(obj_id, pm_dir, gui=False)
    gt_model = GTFlowModel(raw_data, env)

    env.reset()

    for joint in available_joints:  # Close all joints
        info = p.getJointInfo(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[joint],
            env.render_env.client_id,
        )
        init_angle, target_angle = info[8], info[9]
        env.set_joint_state(joint, init_angle)
        # print(init_angle, target_angle)

    target_link = available_joints[joint_id]
    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    # Make the predictions
    angle_sample_cnts = 100
    trial_cnts = 20
    xs = []
    rmses = []
    cosines = []

    flowbot_xs = []
    flowbot_rmses = []
    flowbot_cosines = []

    past_gt_flow = None
    past_pcd = None

    for angle_ratio, angle in tqdm.tqdm(
        zip(
            np.linspace(0, 100, angle_sample_cnts),
            np.linspace(init_angle, target_angle, angle_sample_cnts),
        )
    ):
        # print(angle_ratio)
        env.set_joint_state(target_link, angle)
        # env.set_joint_state(target_link, target_angle)
        pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

        gt_flow = gt_model(pc_obs)
        # nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=-1) != 0.0)
        nonzero_gt_flowixs = pc_seg == env.render_env.link_name_to_index[target_link]
        gt_flow_nz = gt_flow[nonzero_gt_flowixs]

        # FlowBot
        pred_flow = flowbot(copy.deepcopy(pc_obs))[:, 0, :]
        pred_flow_nz = pred_flow[nonzero_gt_flowixs]
        flowbot_xs.append(angle_ratio)
        rmse, cos_dist, mag_error = flow_metrics(pred_flow_nz, gt_flow_nz)
        flowbot_rmses.append(rmse)
        flowbot_cosines.append(cos_dist)

        # HisPNDiT
        for i in range(trial_cnts):
            with torch.no_grad():
                pred_flow = model(
                    copy.deepcopy(pc_obs),
                    history_pcd=past_pcd,
                    history_flow=past_gt_flow,
                )[:, 0, :]

            pred_flow_nz = pred_flow[nonzero_gt_flowixs]

            xs.append(angle_ratio)
            # cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=-1).mean()
            # ys.append(cos_dist)

            rmse, cos_dist, mag_error = flow_metrics(pred_flow_nz, gt_flow_nz)
            rmses.append(rmse)
            cosines.append(cos_dist)

        past_gt_flow = gt_flow.cpu().numpy()
        past_pcd = P_world

    import matplotlib.pyplot as plt

    # ----------Flowbot-------------
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(flowbot_xs, flowbot_cosines, c="gray", s=20, edgecolors="black")
    plt.title(f"Door {obj_id}")
    plt.xlabel("Open ratio (%)")
    plt.ylabel("Cosine similarity")
    # plt.ylabel('Mag error (%)')

    plt.savefig(f"./angle_visuals/{obj_id}_{joint_id}_cos_flowbot.jpg")
    plt.clf()

    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(flowbot_xs, flowbot_rmses, c="gray", s=20, edgecolors="black")
    plt.title(f"Door {obj_id}")
    plt.xlabel("Open ratio (%)")
    plt.ylabel("RMSE")
    # plt.ylabel('Mag error (%)')

    plt.savefig(f"./angle_visuals/{obj_id}_{joint_id}_rmse_flowbot.jpg")
    plt.clf()

    # ----------HisPNDiT-------------
    plt.gcf().set_facecolor("none")
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(xs, cosines, c="orange", s=20, edgecolors="red", alpha=0.2)
    plt.title(f"Door {obj_id}")
    plt.xlabel("Open ratio (%)")
    plt.ylabel("Cosine similarity")
    # plt.ylabel('Mag error (%)')

    plt.savefig(f"./angle_visuals/{obj_id}_{joint_id}_cos_flowbothd.jpg")
    plt.clf()

    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(xs, rmses, c="orange", s=20, edgecolors="red", alpha=0.2)
    plt.title(f"Door {obj_id}")
    plt.xlabel("Open ratio (%)")
    plt.ylabel("RMSE")
    # plt.ylabel('Mag error (%)')

    plt.savefig(f"./angle_visuals/{obj_id}_{joint_id}_rmse_flowbothd.jpg")
    plt.clf()

    breakpoint()
