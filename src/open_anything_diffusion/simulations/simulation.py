# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
import rpad.pyg.nets.pointnet2 as pnp
import torch
from rpad.partnet_mobility_utils.data import PMObject

from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
from open_anything_diffusion.simulations.suction import (  # compute_flow,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
)


def trial_flow(
    obj_id="41083",
    n_steps=30,
    all_joint=True,
    available_joints=None,
    gui=False,
    website=False,
    pm_dir=os.path.expanduser("~/datasets/partnet-mobility/raw"),
):
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        model = GTFlowModel(raw_data, env)
        fig, result = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
        )
        if result.assertion is False:
            with open(
                "/home/yishu/open_anything_diffusion/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results


# Trial with groundtruth trajectories
def trial_gt_trajectory(
    obj_id="41083",
    traj_len=10,
    n_steps=30,
    all_joint=True,
    available_joints=None,
    gui=False,
    website=False,
):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        # model = GTFlowModel(raw_data, env)
        model = GTTrajectoryModel(raw_data, env, traj_len)
        fig, result = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
        )
        if result.assertion is False:
            with open(
                "/home/yishu/open_anything_diffusion/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results

    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    # raw_data = PMObject(os.path.join(pm_dir, obj_id))

    # available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
    #     "slider"
    # )

    # joint = available_joints[np.random.randint(0, len(available_joints))]
    # model = GTTrajectoryModel(raw_data, env, traj_len)

    # # t0 = time.perf_counter()
    # print(f"opening {joint.name}, {joint.label}")
    # run_trial(env, raw_data, joint.name, model, n_steps=1)


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
    obj_id="41083",
    network=None,
    n_step=1,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)

    model = FlowSimulationInferenceModule(network)

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        gt_model = GTFlowModel(raw_data, env)
        fig, result = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            gt_model=gt_model,
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
        )
        if result.assertion is False:
            with open(
                "/home/yishu/open_anything_diffusion/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results


if __name__ == "__main__":
    np.random.seed(42)
    # trial_flow(obj_id="41083", available_joints=["link_0"], gui=True, website=False)
    # trial_gt_trajectory(obj_id="35059", traj_len=15, gui=True)
    # trial_with_prediction(obj_id="35059", traj_len=15, n_step=1, gui=True)

    # length = 15
    # network_15 = create_network(
    #     traj_len=15,
    #     ckpt_file="/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-19/14-51-22/checkpoints/epoch=94-step=74670-val_loss=0.00-weights-only.ckpt",
    # )

    length = 1
    network_1 = create_network(
        traj_len=1,
        ckpt_file="/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-18/23-52-34/checkpoints/epoch=77-step=61308-val_loss=0.00-weights-only.ckpt",
    )
    figs, trial_results = trial_with_prediction(
        obj_id="102044", network=network_1, n_step=15, gui=False, all_joint=True
    )
    print(trial_results)

    # figs[list(figs.keys())[0]].show()
    # trial_with_prediction(obj_id="35059", network=network_15, n_step=1, gui=False, all_joint=False)
