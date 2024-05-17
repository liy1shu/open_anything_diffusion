# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
import pybullet as p
import rpad.pyg.nets.pointnet2 as pnp
import torch
from rpad.partnet_mobility_utils.data import PMObject

from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder
from open_anything_diffusion.simulations.suction import (  # compute_flow,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
    run_trial_with_history,
)


def trial_flow(
    obj_id="41083",
    n_steps=30,
    all_joint=True,
    available_joints=None,
    gui=False,
    website=False,
    pm_dir=os.path.expanduser("~/datasets/partnet-mobility/convex"),
    # pm_dir=os.path.expanduser("~/datasets/partnet-mobility/raw"),
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

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        sim_trajectories.append(sim_trajectory)
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

    return figs, results, sim_trajectories


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
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
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

    sim_trajectories = []
    results = []
    movable_links = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

        # Close all joints:
        for link_to_restore in [
            joint.name
            for joint in raw_data.semantics.by_type("hinge")
            + raw_data.semantics.by_type("slider")
        ]:
            info = p.getJointInfo(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[link_to_restore],
                env.render_env.client_id,
            )
            init_angle, target_angle = info[8], info[9]
            env.set_joint_state(link_to_restore, init_angle)

        # model = GTFlowModel(raw_data, env)
        model = GTTrajectoryModel(raw_data, env, traj_len)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        # raw_data = PMObject(os.path.join(pm_dir, obj_id))
        sim_trajectories.append(sim_trajectory)
        if result.success:
            movable_links.append(joint_name)
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

    return figs, results, movable_links, sim_trajectories

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
    gt_mask=False,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)

    model = FlowSimulationInferenceModule(network, mask_input_channel=gt_mask)

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        gt_model = GTFlowModel(raw_data, env) if gt_mask else None

        # Close all joints:
        for link_to_restore in [
            joint.name
            for joint in raw_data.semantics.by_type("hinge")
            + raw_data.semantics.by_type("slider")
        ]:
            info = p.getJointInfo(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[link_to_restore],
                env.render_env.client_id,
            )
            init_angle, target_angle = info[8], info[9]
            env.set_joint_state(link_to_restore, init_angle)

        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            gt_model=gt_model,
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        sim_trajectories.append(sim_trajectory)
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

    return figs, results, sim_trajectories


def trial_with_diffuser(
    obj_id="41083",
    model=None,
    n_step=30,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)
    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

        # Close all joints:
        for link_to_restore in [
            joint.name
            for joint in raw_data.semantics.by_type("hinge")
            + raw_data.semantics.by_type("slider")
        ]:
            info = p.getJointInfo(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[link_to_restore],
                env.render_env.client_id,
            )
            init_angle, target_angle = info[8], info[9]
            env.set_joint_state(link_to_restore, init_angle)

        # gt_model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            gt_model=None,  # Don't need mask
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        sim_trajectories.append(sim_trajectory)
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

    return figs, results, sim_trajectories


def trial_with_diffuser_history(
    obj_id="41083",
    model=None,
    history_model=None,
    n_step=30,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)
    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

        # Close all joints:
        for link_to_restore in [
            joint.name
            for joint in raw_data.semantics.by_type("hinge")
            + raw_data.semantics.by_type("slider")
        ]:
            info = p.getJointInfo(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[link_to_restore],
                env.render_env.client_id,
            )
            init_angle, target_angle = info[8], info[9]
            env.set_joint_state(link_to_restore, init_angle)

        # gt_model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial_with_history(
            env,
            raw_data,
            joint_name,
            model,
            history_model,
            gt_model=None,  # Don't need mask
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        sim_trajectories.append(sim_trajectory)
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

    return figs, results, sim_trajectories


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    # trial_flow(obj_id="41083", available_joints=["link_0"], gui=True, website=False)
    # trial_gt_trajectory(obj_id="8867", traj_len=30, gui=False)
    # trial_with_prediction(obj_id="35059", traj_len=15, n_step=1, gui=True)

    # length = 15
    # network_15 = create_network(
    #     traj_len=15,
    #     ckpt_file="/home/yishu/open_anything_diffusion/scripts/logs/train_flowbot/2023-07-19/14-51-22/checkpoints/epoch=94-step=74670-val_loss=0.00-weights-only.ckpt",
    # )

    # # length = 1
    # network_1 = pnp.PN2Dense(
    #     in_channels=0,
    #     out_channels=3,
    #     p=pnp.PN2DenseParams(),
    # )
    # ckpt = torch.load("/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_flowbot.ckpt")
    # network_1.load_state_dict(
    #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    # )
    # network_1.eval()
    # # network_1.load_state_dict(torch.load()["state_dict"])
    # trial_figs, trial_results, sim_trajectory = trial_with_prediction(
    #     obj_id="102358", network=network_1, n_step=30, gui=False, website=True, all_joint=True
    # )
    # print(trial_results)

    # figs[list(figs.keys())[0]].show()
    # trial_with_prediction(obj_id="35059", network=network_15, n_step=1, gui=False, all_joint=False)

    # Trial with dit
    from open_anything_diffusion.models.modules.dit_models import DiT

    torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    np.set_printoptions(precision=10)

    network = DiT(
        in_channels=3 + 3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        learn_sigma=True,
    ).cuda()

    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-10/10-54-13/checkpoints/epoch=459-step=80500-val_loss=0.00-weights-only.ckpt"
    ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt"
    # ckpt_file = "/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_flowbot.ckpt"
    from hydra import compose, initialize

    initialize(config_path="../../../configs", version_base="1.3")
    cfg = compose(config_name="eval_sim")

    from open_anything_diffusion.models.flow_diffuser_dit import (
        FlowTrajectoryDiffuserSimulationModule_DiT,
    )
    from open_anything_diffusion.models.flow_diffuser_hisdit import (
        FlowTrajectoryDiffuserSimulationModule_HisDiT,
    )

    model = FlowTrajectoryDiffuserSimulationModule_DiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()

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
    history_model = FlowTrajectoryDiffuserSimulationModule_HisDiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    history_model.load_from_ckpt(ckpt_file)
    history_model.eval()

    # trial_figs, trial_results, sim_trajectory = trial_with_diffuser(
    trial_figs, trial_results, sim_trajectory = trial_with_diffuser_history(
        # obj_id="8877",
        obj_id="8877",
        model=history_model,
        history_model=history_model,
        n_step=30,
        gui=False,
        website=cfg.website,
        all_joint=False,
        available_joints=["link_1"],
    )
