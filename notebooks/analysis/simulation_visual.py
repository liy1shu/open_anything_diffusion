import numpy as np
import torch

from open_anything_diffusion.simulations.simulation import *

np.random.seed(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)
from hydra import compose, initialize

initialize(config_path="../../configs", version_base="1.3")
cfg = compose(config_name="eval_sim_switch")

from open_anything_diffusion.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserSimulationModule_DiT,
)
from open_anything_diffusion.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffuserSimulationModule_PNDiT,
)
from open_anything_diffusion.models.modules.dit_models import DiT, PN2DiT, PN2HisDiT
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder

model_one_name = "DiT"
model_two_name = "HisPNDiT"


# Trial with dit
if model_one_name == "DiT":
    network = DiT(
        in_channels=3 + 3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        # depth=12,
        # hidden_size=384,
        # num_heads=6,
        learn_sigma=True,
    ).cuda()
    ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt"
    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-05-02/12-35-27/checkpoints/epoch=109-step=243100-val_loss=0.00-weights-only.ckpt"

    model = FlowTrajectoryDiffuserSimulationModule_DiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()
elif model_one_name == "PNDiT":
    network = PN2DiT(
        in_channels=3,
        depth=5,
        hidden_size=128,
        patch_size=1,
        num_heads=4,
        n_points=1200,
    ).cuda()
    ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_pndit/2024-04-23/05-01-44/checkpoints/epoch=469-step=1038700-val_loss=0.00-weights-only.ckpt"
    model = FlowTrajectoryDiffuserSimulationModule_PNDiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()


if model_two_name == "HisDiT":
    from open_anything_diffusion.models.flow_diffuser_hisdit import (
        FlowTrajectoryDiffuserSimulationModule_HisDiT,
    )

    # History model
    network = {
        "DiT": DiT(
            in_channels=3 + 3 + 128,
            depth=5,
            hidden_size=128,
            num_heads=4,
            learn_sigma=True,
        ).cuda(),
        "History": HistoryEncoder(
            history_dim=128, history_len=1, batch_norm=False
        ).cuda(),
    }

    ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hisdit/2024-05-10/12-09-08/checkpoints/epoch=439-step=243320-val_loss=0.00-weights-only.ckpt"
    switch_model = FlowTrajectoryDiffuserSimulationModule_HisDiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    switch_model.load_from_ckpt(ckpt_file)
    switch_model.eval()
elif model_two_name == "HisPNDiT":
    from open_anything_diffusion.models.flow_diffuser_hispndit import (
        FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
    )

    # History model
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

    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hispndit/2024-05-25/02-00-54/checkpoints/epoch=299-step=248700-val_loss=0.00-weights-only.ckpt"
    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hispndit/2024-05-25/02-00-54/checkpoints/epoch=359-step=298440.ckpt"
    ckpt_file = (
        "/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_hispndit.ckpt"
    )
    switch_model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    switch_model.load_from_ckpt(ckpt_file)
    switch_model.eval()
elif model_two_name == "pn++":
    import rpad.pyg.nets.pointnet2 as pnp_orig

    from open_anything_diffusion.models.flow_trajectory_predictor import (
        FlowSimulationInferenceModule,
    )

    network = pnp_orig.PN2Dense(
        in_channels=0,
        out_channels=3,
        p=pnp_orig.PN2DenseParams(),
    ).cuda()
    switch_model = FlowSimulationInferenceModule(
        network, cfg.switch_inference, cfg.switch_model
    ).cuda()
    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-03-30/08-16-05/checkpoints/epoch=88-step=98345-val_loss=0.00-weights-only.ckpt"
    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-05-25/04-17-41/checkpoints/epoch=95-step=53088-val_loss=0.00-weights-only.ckpt"
    ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-05-26/02-37-08/checkpoints/epoch=98-step=109395-val_loss=0.00-weights-only.ckpt"
    switch_model.load_from_ckpt(ckpt_file)
    switch_model.eval()


# obj_ids = ["8877", "8877", "9065", "8867", "8893"]
# joint_ids = [0, 1, 0, 0, 1]
obj_ids = ["8867"]
joint_ids = [0]
pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")

for obj_id, joint_id in zip(obj_ids, joint_ids):
    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )
    available_joints = [joint.name for joint in available_joints]
    target_link = available_joints[joint_id]
    print(target_link)

    # trial_figs, trial_results, sim_trajectory = trial_with_diffuser(
    trial_figs, trial_results, all_signals = trial_with_diffuser_history(
        # obj_id="8877",
        obj_id=obj_id,
        model=switch_model,
        history_model=switch_model,
        # model=switch_model,
        # model=model,
        # switch_model=switch_model,
        # switch_model=model,
        # history_for_models=[False, True],
        n_step=30,
        gui=False,
        website=True,
        all_joint=False,
        available_joints=[target_link],
        consistency_check=True,
        history_filter=True,
        analysis=True
        # return_switch_ids=True,
    )
    (sim_trajectory, update_history_signals, cc_cnts, sgp_signals) = all_signals[0]
    # breakpoint()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # plt.title(f'DiT & FowBot - Door {obj_id} {target_link}')
    # plt.title(f'DiT - Door {obj_id} {target_link}')
    # plt.title(f'Flowbot - Door {obj_id} {target_link}')
    # plt.title(f'PNDiT & FlowBot - {obj_id} {target_link}')
    plt.title(f"HisPNDiT - {obj_id} {target_link}")
    fig, ax1 = plt.subplots()

    x = [i for i in range(31)]
    y = sim_trajectory
    # colors = ["red" if color else "blue" for color in colors[1:]]
    colors = ["blue"] * 30
    # colors = ["red"] * 30

    for i in range(len(x) - 1):
        plt.plot(x[i : i + 2], y[i : i + 2], color=colors[i], alpha=0.6)

    plt.xlabel("Step")
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylabel("Open ratio")

    for i in range(len(update_history_signals)):
        if update_history_signals[i]:
            plt.plot(x[i], y[i], marker="*", color="red", markersize=10, alpha=0.8)
        if sgp_signals[i]:
            plt.plot(x[i], y[i], marker="^", color="yellow", markersize=10, alpha=0.8)

    new_cc_cnts = [0] * len(x)
    for i in range(len(cc_cnts)):
        new_cc_cnts[i] = cc_cnts[i] + 1

    ax2 = ax1.twinx()
    # Plotting the second dataset
    ax2.hist(new_cc_cnts, bins=x, color="blue", alpha=0.2)
    ax2.set_ylabel("Trial counts", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # plt.savefig(f'./traj_visuals/{obj_id}_{target_link}_dit&flowbot.jpg')
    # plt.savefig(f'./traj_visuals/{obj_id}_{target_link}_dit.jpg')
    # plt.savefig(f'./traj_visuals/{obj_id}_{target_link}_flowbot.jpg')
    # plt.savefig(f'./traj_visuals/{obj_id}_{target_link}_pndit&flowbot.jpg')
    plt.savefig(f"./traj_visuals/{obj_id}_{target_link}_hispndit.jpg")
