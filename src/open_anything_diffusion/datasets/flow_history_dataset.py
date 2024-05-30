import math
import random
from typing import List, Optional, Protocol, Union, cast

import flowbot3d.datasets.flow_dataset as f3dd
import numpy as np
import pybullet as p
import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer
from torch_geometric.data import Data

"""
Variable length history dataset
- Generated by taking the joint limits of a randomly selected joint
- Rendering small change in movement for 100 steps
- Randomly selecting a variable length (K) from the history
"""
############################################################

# Joints helper functions


def get_random_joint(obj_id, client_id, seed=None, raw_data_obj=None):
    rng = np.random.default_rng(seed)
    n_joints = p.getNumJoints(obj_id, client_id)
    articulation_ixs = []
    for joint_ix in range(n_joints):
        jinfo = p.getJointInfo(obj_id, joint_ix, client_id)
        if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
            joint_name = jinfo[1].decode("UTF-8")
            joint_type = int(jinfo[2])
            start_end = raw_data_obj.get_joint(joint_name).limit
            if start_end is not None:
                start, end = start_end
                if start >= end:
                    continue
            articulation_ixs.append((joint_name, joint_type, joint_ix))
    selected_ix = articulation_ixs[rng.choice(len(articulation_ixs))]
    joint_name, joint_type, joint_ix = selected_ix
    return joint_name, joint_type, joint_ix


def get_joint_type(obj_id, client_id, joint_ix):
    jinfo = p.getJointInfo(obj_id, joint_ix, client_id)
    if jinfo[2] == p.JOINT_REVOLUTE:
        return "R"
    elif jinfo[2] == p.JOINT_PRISMATIC:
        return "P"


def get_joints(obj_id, client_id):
    joints = []
    for i in range(p.getNumJoints(obj_id, client_id)):
        jinfo = p.getJointInfo(obj_id, i, client_id)
        joints.append(jinfo)
    return joints


def get_joint_angles(obj_id, client_id):
    angles = {}
    for i in range(p.getNumJoints(obj_id, client_id)):
        jinfo = p.getJointInfo(obj_id, i, client_id)
        jstate = p.getJointState(obj_id, i, client_id)
        angles[jinfo[12].decode("UTF-8")] = jstate[0]
    return angles


############################################################


class FlowHistory(Protocol):
    id: str  # Object ID.
    # curr_pos: torch.Tensor  # Points in the point cloud.
    pos: torch.Tensor
    point: torch.Tensor
    delta: torch.Tensor  # instantaneous positive 3D flow.
    mask: torch.Tensor  # Mask of the part of interest.
    K: int  # Size of history window (i.e. number of point clouds in the history)
    history: torch.Tensor  # Array of K point clouds which form the history of the observation


class FlowHistoryDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        trajectory_len: int = 1,
        history_len: int = 1,
        special_req: str = None,
        n_points: Optional[int] = 1200,
        seed: int = 42,
    ):
        super().__init__()

        self.seed = seed
        self._dataset = pmd.PCDataset(root=root, split=split, renderer="pybullet")

        self.randomize_joints = randomize_joints
        self.randomize_camera = randomize_camera
        self.trajectory_len = trajectory_len
        self.history_len = history_len
        self.special_req = special_req
        self.n_points = n_points

    def len(self) -> int:
        return len(self._dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self._dataset._ids[index])

    @staticmethod
    def get_processed_dir(
        randomize_joints,
        randomize_camera,
        trajectory_len,
        history_len,
        special_req=None,
        toy_dataset_id=None,
    ):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        # breakpoint()
        if special_req is None and toy_dataset_id is None:
            return f"processed_history_{trajectory_len}_{history_len}_{joint_chunk}_{camera_chunk}_random"
        elif special_req is not None and toy_dataset_id is None:
            # fully_closed
            # half_half
            return f"processed_history_{trajectory_len}_{history_len}_{joint_chunk}_{camera_chunk}_{special_req}"
        elif special_req is None and toy_dataset_id is not None:
            # fully_closed
            # half_half
            return f"processed_history_{trajectory_len}_{history_len}_{joint_chunk}_{camera_chunk}_toy{toy_dataset_id}_random"
        else:
            return f"processed_history_{trajectory_len}_{history_len}_{joint_chunk}_{camera_chunk}_{special_req}_toy{toy_dataset_id}"

    def get_data(self, obj_id: str, seed=None) -> FlowHistory:
        # Initial randomization parameters.
        # Select the camera.
        this_sample_open = True
        if self.special_req is None:
            joints = "random" if self.randomize_joints else None
        elif self.special_req == "half-half":
            this_sample_open = random.randint(0, 100) < 50
            # print(this_sample_open)
            if this_sample_open:  # Open this sample - with history
                joints = "random"
            else:  # Close this sample - without history
                joints = "fully-closed"
        elif self.special_req == "50-150":
            this_sample_open = random.randint(0, 150) < 50
            # print(this_sample_open)
            if this_sample_open:  # Open this sample - with history
                joints = "random"
            else:  # Close this sample - without history
                joints = "fully-closed"
        elif self.special_req == "fully-closed":
            this_sample_open = False
            joints = "fully-closed"
        else:
            assert True, f"{self.special_req} mode not supported in history dataset."
        camera_xyz = "random" if self.randomize_camera else None

        rng = np.random.default_rng(seed)
        seed1, seed2, seed3, seed4 = rng.bit_generator._seed_seq.spawn(4)  # type: ignore
        _ = self._dataset.get(  # Just to create a renderer
            obj_id=obj_id,
            joints=joints,
            camera_xyz=camera_xyz,
            seed=seed1,
        )

        raw_data_obj = self._dataset.pm_objs[obj_id].obj
        # Randomly select a joint to modify by poking through the guts.
        renderer: PybulletRenderer = self._dataset.renderers[obj_id]  # type: ignore
        (
            joint_name,
            joint_type,
            joint_ix,
        ) = get_random_joint(  # Choose a joint to manipulate with
            obj_id=renderer._render_env.obj_id,
            client_id=renderer._render_env.client_id,
            seed=seed2,
            raw_data_obj=raw_data_obj,
        )
        # print(joint_name, "open" if this_sample_open else "close")
        data_t0 = (
            self._dataset.get(  # Re-render the object with one specific random joint!
                obj_id=obj_id,
                joints=joints,
                camera_xyz=camera_xyz,
                seed=seed1,
                random_joint_id=joint_name,
            )
        )
        pos_t0 = data_t0["pos"]

        # Compute the flow + mask at that time.
        # target_point_t0, _, flow_t0 = compute_normalized_flow(
        flow_t0 = f3dd.compute_normalized_flow(
            P_world=pos_t0,
            T_world_base=data_t0["T_world_base"],
            current_jas=data_t0["angles"],
            pc_seg=data_t0["seg"],
            labelmap=data_t0["labelmap"],
            pm_raw_data=self._dataset.pm_objs[obj_id],
            linknames="all",
        )
        mask_t0 = (~(flow_t0 == 0.0).all(axis=-1)).astype(int)

        # Compute the states for the camera and joints at t1.
        # Camera should be the same.
        camera_xyz_t1 = data_t0["T_world_cam"][:3, 3]
        joints_t0 = data_t0["angles"]

        # print("t0:", joints_t0)

        ###################################################################

        
        if self.history_len > 1:
            K = random.randint(0, self.history_len)
            # print(K)
            # K = 0
        else:
            K = (
                0 if not this_sample_open else 1
            )  # If fully closed - then return None history
            # print(K)
        # print("K", K)
        d_theta = 0
        if this_sample_open:  # Pick a joint to open
            joint = raw_data_obj.get_joint(joint_name)

            if joint.limit == None:  # revolute free moving
                min_theta, max_theta = 0, 2 * math.pi
            else:
                min_theta, max_theta = joint.limit

            assert (
                max_theta > min_theta
            ), "Selected a joint with min theta >= max theta?"

            interval_cnt = random.randint(5, 50)
            d_theta = (max_theta - min_theta) / interval_cnt

        # HACK HACK HACK we need to make sure that the joint is actually in the joint list.
        # This is a bug in the underlying library, annoying.
        joints_t1 = {
            jn: jv
            for jn, jv in joints_t0.items()
            if jn in renderer._render_env.jn_to_ix
        }

        ###################################################################
        # Render. and compute values.
        lengths = []

        # end_theta = min_theta + end_ix * d_theta

        step_id = 0
        while step_id <= K:
            # print("t1:", joints_t1)
            # Describe the action that was taken.
            action = np.zeros(len(joints_t0))
            action[joint_ix] = d_theta

            # Get the second render.
            data_t1 = self._dataset.get(
                obj_id=obj_id, joints=joints_t1, camera_xyz=camera_xyz_t1, seed=seed3
            )
            pos_t1 = data_t1["pos"]

            # Compute the flow + mask at that time.
            flow_t1 = f3dd.compute_normalized_flow(
                # target_point_t1, _, flow_t1 = compute_normalized_flow(
                P_world=pos_t1,
                T_world_base=data_t1["T_world_base"],
                current_jas=data_t1["angles"],
                pc_seg=data_t1["seg"],
                labelmap=data_t1["labelmap"],
                pm_raw_data=self._dataset.pm_objs[obj_id],
                linknames="all",
            )

            mask_t1 = (~(flow_t1 == 0.0).all(axis=-1)).astype(int)

            # Downsample.
            if self.n_points:
                rng = np.random.default_rng(seed4)

                ixs_t0 = rng.permutation(range(len(pos_t0)))[: self.n_points]
                pos_t0 = pos_t0[ixs_t0]
                flow_t0 = flow_t0[ixs_t0]
                # target_point_t0 = target_point_t0[ixs_t0]
                mask_t0 = mask_t0[ixs_t0]

                ixs_t1 = rng.permutation(range(len(pos_t1)))[: self.n_points]
                pos_t1 = pos_t1[ixs_t1]
                flow_t1 = flow_t1[ixs_t1]
                # target_point_t1 = target_point_t1[ixs_t1]
                mask_t1 = mask_t1[ixs_t1]

            # Add to series of history
            if step_id == 0:
                # history = np.array([pos_t0])
                # flow_history = np.array([flow_t0])
                # target_point_history = np.array([target_point_t0])
                # lengths.append(len(pos_t0))
                history = np.array([pos_t1])
                flow_history = np.array([flow_t1])
                # target_point_history = np.array([target_point_t1])
                lengths.append(len(pos_t1))
            else:
                history = np.append(history, [pos_t1], axis=0)
                flow_history = np.append(flow_history, [flow_t1], axis=0)
                # target_point_history = np.append(target_point_history, [target_point_t1], axis=0)
                lengths.append(len(pos_t1))

            step_id += 1

            # Rotate the target joint for an angle
            joints_t1[joint_name] += d_theta

        # start_ix = random.randint(0, num_obs - 2) # min 1 observation
        # temp_K = random.randint(1, 20)
        # end_ix = min(start_ix + temp_K, num_obs - 1) # upperbound of num_obs-1
        # K = end_ix - start_ix

        # print(step_id)
        curr_pos = history[-1]
        flow = flow_history[-1]

        # history = (
        #     history[:-1] if K >= 1 else history * 0
        # )  # No history, but the shape should be the same
        flow_history = flow_history[:-1] if K >= 1 else flow_history * 0
        lengths = lengths[:-1] if K >= 1 else lengths
        # print(history.shape, flow_history.shape, lengths)

        history = history.reshape(-1, history.shape[-1])  #
        flow_history = flow_history.reshape(-1, flow_history.shape[-1])
        # target_point_history = target_point_history.reshape(-1, target_point_history.shape[-1])
        data = Data(
            id=obj_id,
            num_points=torch.tensor([curr_pos.shape[0]]),  # N: shape of point cloud
            pos=torch.from_numpy(curr_pos).float(),
            action=torch.from_numpy(action).float(),
            delta=torch.from_numpy(flow).unsqueeze(1).float(),
            # point=torch.from_numpy(target_point).unsqueeze(1).float(),
            mask=torch.from_numpy(mask_t1).float(),
            history=torch.from_numpy(history).float(),  # N*K, 3
            flow_history=torch.from_numpy(  # N*K, 3
                flow_history
            ).float(),  # Snapshot of flow history
            # link=joint.child,  # child of the joint gives you the link that the joint is connected to
            K=K,  # length of history
            lengths=torch.as_tensor(lengths).int(),  # size of point cloud
        )

        return cast(FlowHistory, data)