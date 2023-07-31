# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
import rpad.pyg.nets.pointnet2 as pnp
import torch

from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
from open_anything_diffusion.simulations.pm_raw import PMRawData
from open_anything_diffusion.simulations.suction import (  # compute_flow,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
)


def trial_flow(obj_id="41083", gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMRawData(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = GTFlowModel(raw_data, env)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    run_trial(env, raw_data, joint.name, model, n_steps=10)


# Trial with groundtruth trajectories
def trial_gt_trajectory(obj_id="41083", traj_len=10, gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMRawData(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = GTTrajectoryModel(raw_data, env, traj_len)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    run_trial(env, raw_data, joint.name, model, n_steps=1)


def create_network(traj_len=15, ckpt_file=None):
    network = pnp.PN2Dense(
        in_channels=1, out_channels=3 * traj_len, p=pnp.PN2DenseParams()
    )
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    return network


# Trial with model predicted trajectories
def trial_with_prediction(
    obj_id="41083", network=None, n_step=1, gui=False, all_joint=False
):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    raw_data = PMRawData(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    model = FlowSimulationInferenceModule(network)

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    results = []
    figs = {}
    for joint in picked_joints:
        # t0 = time.perf_counter()
        print(f"opening {joint.name}, {joint.label}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        fig, result = run_trial(env, raw_data, joint.name, model, n_steps=n_step)
        figs[joint.name] = fig
        results.append(result)

    return figs, results


if __name__ == "__main__":
    np.random.seed(42)
    # trial_flow(obj_id="41083", gui=True)
    # trial_gt_trajectory(obj_id="35059", traj_len=15, gui=True)
    # trial_with_prediction(obj_id="35059", traj_len=15, n_step=1, gui=True)

    # length = 15
    network_15 = create_network(
        traj_len=15,
        ckpt_file="/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-19/14-51-22/checkpoints/epoch=94-step=74670-val_loss=0.00-weights-only.ckpt",
    )
    # length = 1
    # network_1 = create_network(traj_len=15, ckpt_file="/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-18/23-52-34/checkpoints/epoch=77-step=61308-val_loss=0.00-weights-only.ckpt")
    figs, trial_results = trial_with_prediction(
        obj_id="41083", network=network_15, n_step=1, gui=True, all_joint=False
    )
    print(trial_results)
    figs[list(figs.keys())[0]].show()
    # trial_with_prediction(obj_id="35059", network=network_15, n_step=1, gui=False, all_joint=False)
