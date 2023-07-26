import copy
from functools import reduce
from typing import List, Sequence

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from open_anything_diffusion.simulations.pm_raw import Joint, PMRawData


def fk(chain: List[Joint], joint_angles: Sequence[float]) -> np.ndarray:
    def compute_T_link_childnew(joint: Joint, angle: float) -> np.ndarray:
        T_link_child: np.ndarray = np.eye(4)
        if joint.origin is not None:
            xyz, rpy = joint.origin
            T_link_child[:3, 3] = xyz
            T_link_child[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

        T_articulation: np.ndarray = np.eye(4)
        if joint.type == "revolute" or joint.type == "continuous":
            theta = angle
            if theta != 0.0:
                axis = (
                    joint.axis
                    if joint.axis is not None
                    else np.asarray([1.0, 0.0, 0.0])
                )
                R = trimesh.transformations.rotation_matrix(theta, axis)
                T_articulation[:3, :3] = R[:3, :3]

        elif joint.type == "prismatic":
            theta = angle
            axis = joint.axis if joint.axis is not None else np.asarray([1.0, 0.0, 0.0])
            axis = axis / np.linalg.norm(axis)

            T_articulation[:3, 3] = axis * theta

        T_link_childnew: np.ndarray = T_link_child @ T_articulation
        return T_link_childnew

    tforms = [compute_T_link_childnew(j, a) for j, a in zip(chain, joint_angles)]

    # Compose all transforms into a single one. Basically, we left-apply each transform
    # down the chain.
    T_base_endlink = reduce(lambda T_gp_p, T_p_c: T_gp_p @ T_p_c, tforms, np.eye(4))  # type: ignore

    return T_base_endlink


def compute_new_points(
    P_world_pts: np.ndarray,
    T_world_base: np.ndarray,
    kinematic_chain: List[Joint],
    current_ja: Sequence[float],
    target_ja: Sequence[float],
) -> np.ndarray:
    if P_world_pts.shape == (3,):
        P_world_pts = np.expand_dims(P_world_pts, 0)

    # Validation.
    assert len(P_world_pts.shape) == 2
    assert P_world_pts.shape[1] == 3
    assert len(kinematic_chain) == len(current_ja) == len(target_ja)
    assert T_world_base.shape == (4, 4)

    N = len(P_world_pts)
    Ph_world_pts = np.concatenate([P_world_pts, np.ones((N, 1))], axis=1)

    ################## STEP 1. ####################
    # Put all the points in the frame of the final joint.

    # Do forward kinematics to get the position of the final link.
    T_base_endlink = fk(kinematic_chain, current_ja)
    T_world_endlink = T_world_base @ T_base_endlink
    T_endlink_world = np.linalg.inv(T_world_endlink)

    # Points in the final link frame.
    Ph_endlink_pts = (T_endlink_world @ Ph_world_pts.T).T
    assert Ph_endlink_pts.shape == (N, 4)

    ################## STEP 2. ####################
    # Compute the frame of the final joint for the new joint angles,
    # and find the relative transform from the original frame.

    T_base_endlinknew = fk(kinematic_chain, target_ja)
    T_endlink_endlinknew = np.linalg.inv(T_base_endlink) @ T_base_endlinknew

    ################## STEP 3. ####################
    # Apply that relative transform to the points in the end link frame,
    # and put it all back in the world frame.
    Ph_endlink_ptsnew = (T_endlink_endlinknew @ Ph_endlink_pts.T).T
    Ph_world_ptsnew = (T_world_endlink @ Ph_endlink_ptsnew.T).T
    assert Ph_world_ptsnew.shape == (N, 4)
    P_world_ptsnew: np.ndarray = Ph_world_ptsnew[:, :3]  # not homogenous anoymore.

    return P_world_ptsnew


def apply_articulation(
    P_world: np.ndarray,
    T_world_base: np.ndarray,
    seg: np.ndarray,
    raw_data: PMRawData,
    joint_states: List[float],
    joint_name: str,
    theta: float,
) -> np.ndarray:
    kinematic_tree = raw_data.obj

    ins_labels = {link.name: i for i, link in enumerate(kinematic_tree.links)}
    ins_map = {i: name for name, i in ins_labels.items()}

    # If we want to actuate a specific joint by name, we find the child
    # link first, and then get the kinematic chain to that link.
    cls = [j.child for j in kinematic_tree.joints if j.name == joint_name]
    assert len(cls) == 1
    child_link = cls[0]
    chain = kinematic_tree.get_chain(link_name=child_link)

    # Filter the current state (which consists of all joints) down to
    # those in our current kinematic tree. Then set the last joint
    # to the desired theta.
    ja_ixs = {j.name: i for i, j in enumerate(kinematic_tree.joints)}
    current_ja = [joint_states[ja_ixs[j.name]] for j in chain]
    target_ja = copy.copy(current_ja)
    target_ja[-1] = theta
    new_joint_states = copy.copy(joint_states)
    new_joint_states[ja_ixs[joint_name]] = theta

    # Select all of the links that are descendents of the child link,
    # so that they may move as well.
    descendent_names = kinematic_tree.children[child_link]
    moving_links = [child_link] + descendent_names
    rev_ins_map = {name: i for i, name in ins_map.items()}
    moving_ins_labels = [rev_ins_map[link] for link in moving_links]

    # Filter down the pointcloud
    new_pos = copy.copy(P_world)
    ixs = np.stack([seg == i for i in moving_ins_labels], axis=0).any(axis=0)
    P_world_childpts = P_world[ixs]

    # Compute the position of the new points.
    P_world_childptsnew = compute_new_points(
        P_world_childpts,
        T_world_base=T_world_base,
        kinematic_chain=chain,
        current_ja=current_ja,
        target_ja=target_ja,
    )
    new_pos[ixs] = P_world_childptsnew

    return new_pos
