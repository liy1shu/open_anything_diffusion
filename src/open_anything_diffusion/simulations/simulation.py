# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
import rpad.pyg.nets.pointnet2 as pnp
import torch

from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
# from open_anything_diffusion.simulations.pm_raw import PMRawData
from rpad.partnet_mobility_utils.data import PMObject
from open_anything_diffusion.simulations.suction import (  # compute_flow,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
)


def trial_flow(obj_id="41083", gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = GTFlowModel(raw_data, env)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    return run_trial(env, raw_data, joint.name, model, n_steps=10)


# Trial with groundtruth trajectories
def trial_gt_trajectory(obj_id="41083", traj_len=10, gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = GTTrajectoryModel(raw_data, env, traj_len)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    run_trial(env, raw_data, joint.name, model, n_steps=1, traj_len=traj_len)


# Trial with model predicted trajectories
def trial_with_prediction(obj_id="41083", traj_len=15, n_step=1, gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    # Load prediction model
    joint = available_joints[np.random.randint(0, len(available_joints))]
    # make predictions
    # filter only needed part
    network = pnp.PN2Dense(
        in_channels=1, out_channels=3 * traj_len, p=pnp.PN2DenseParams()
    )

    # Get the checkpoint file. If it's a wandb reference, download.
    # Otherwise look to disk.
    # checkpoint_reference = cfg.checkpoint.reference
    # if checkpoint_reference.startswith(cfg.wandb.entity):
    #     # download checkpoint locally (if not already cached)
    #     artifact_dir = cfg.wandb.artifact_dir
    #     artifact = run.use_artifact(checkpoint_reference, type="model")
    #     ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    # else:
    #     ckpt_file = checkpoint_reference

    # length = 15
    # ckpt_file = "/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-19/14-51-22/checkpoints/epoch=94-step=74670-val_loss=0.00-weights-only.ckpt"
    # length = 1
    ckpt_file = "/home/yishu/open_anything_backup/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-18/23-52-34/checkpoints/epoch=77-step=61308-val_loss=0.00-weights-only.ckpt"
    # Load the network weights.
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    model = FlowSimulationInferenceModule(network)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    run_trial(env, raw_data, joint.name, model, n_steps=n_step, traj_len=traj_len)


if __name__ == "__main__":
    np.random.seed(42)
    # trial_flow(obj_id="168", gui=False)
    # trial_gt_trajectory(obj_id="35059", traj_len=15, gui=False)
    # trial_with_prediction(obj_id="35059", traj_len=15, n_step=1, gui=False)
    trial_with_prediction(obj_id="35059", traj_len=1, n_step=15, gui=False)
