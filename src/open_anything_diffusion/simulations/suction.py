import copy
import time
from dataclasses import dataclass
from typing import Optional

import imageio
import numpy as np
import pybullet as p
import torch
from flowbot3d.datasets.flow_dataset import compute_normalized_flow
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation
from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper
from scipy.spatial.transform import Rotation as R

from open_anything_diffusion.datasets.flow_trajectory_dataset import (
    compute_flow_trajectory,
)
from open_anything_diffusion.metrics.trajectory import normalize_trajectory


class PMSuctionSim:
    def __init__(self, obj_id: str, dataset_path: str, gui: bool = False):
        self.render_env = PMRenderEnv(obj_id=obj_id, dataset_path=dataset_path, gui=gui)
        self.gui = gui
        self.gripper = FloatingSuctionGripper(self.render_env.client_id)
        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )

    # def run_demo(self):
    #     while True:
    #         self.gripper.set_velocity([0.4, 0, 0.0], [0, 0, 0])
    #         for i in range(10):
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)
    #         contact = self.gripper.detect_contact()
    #         if contact:
    #             break

    #     print("stopping gripper")

    #     self.gripper.set_velocity([0.001, 0, 0.0], [0, 0, 0])
    #     for i in range(10):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)
    #         contact = self.gripper.detect_contact()
    #         print(contact)

    #     print("starting activation")

    #     self.gripper.activate()

    #     self.gripper.set_velocity([0, 0, 0.0], [0, 0, 0])
    #     for i in range(100):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    #     # print("releasing")
    #     # self.gripper.release()

    #     print("starting motion")
    #     for i in range(100):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    #     for _ in range(20):
    #         for i in range(100):
    #             self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
    #             self.gripper.apply_force([-500, 0, 0])
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)

    #         for i in range(100):
    #             self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
    #             self.gripper.apply_force([-500, 0, 0])
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)

    #     print("releasing")
    #     self.gripper.release()

    #     for i in range(1000):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    def reset(self):
        pass

    def reset_gripper(self):
        # print(self.gripper.contact_const)
        assert self.gripper.contact_const, "What?????"
        self.gripper.release()
        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )

    def set_gripper_pose(self, pos, ori):
        self.gripper.set_pose(pos, ori)

    def set_joint_state(self, link_name: str, value: float):
        p.resetJointState(
            self.render_env.obj_id,
            self.render_env.link_name_to_index[link_name],
            value,
            0.0,
            self.render_env.client_id,
        )

    def render(self, filter_nonobj_pts: bool = False, n_pts: Optional[int] = None):
        output = self.render_env.render()
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = output

        if filter_nonobj_pts:
            pc_seg_obj = np.ones_like(pc_seg) * -1
            for k, (body, link) in segmap.items():
                if body == self.render_env.obj_id:
                    ixs = pc_seg == k
                    pc_seg_obj[ixs] = link

            is_obj = pc_seg_obj != -1
            P_cam = P_cam[is_obj]
            P_world = P_world[is_obj]
            pc_seg = pc_seg_obj[is_obj]
        if n_pts is not None:
            perm = np.random.permutation(len(P_world))[:n_pts]
            P_cam = P_cam[perm]
            P_world = P_world[perm]
            pc_seg = pc_seg[perm]

        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap

    def set_camera(self):
        pass

    def teleport_and_approach(
        self, point, contact_vector, video_writer=None, standoff_d: float = 0.2
    ):
        # Normalize contact vector.
        contact_vector = (contact_vector / contact_vector.norm(dim=-1)).float()

        p_teleport = (torch.from_numpy(point) + contact_vector * standoff_d).float()

        # breakpoint()

        e_z_init = torch.tensor([0, 0, 1.0]).float()
        e_y = -contact_vector
        e_x = torch.cross(-contact_vector, e_z_init)
        e_x = e_x / e_x.norm(dim=-1)
        e_z = torch.cross(e_x, e_y)
        e_z = e_z / e_z.norm(dim=-1)
        R_teleport = torch.stack([e_x, e_y, e_z], dim=1)
        R_gripper = torch.as_tensor(
            [
                [1, 0, 0],
                [0, 0, 1.0],
                [0, -1.0, 0],
            ]
        )
        # breakpoint()
        o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()

        self.gripper.set_pose(p_teleport, o_teleport)

        contact = self.gripper.detect_contact(self.render_env.obj_id)
        max_steps = 500
        curr_steps = 0
        self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
        while not contact and curr_steps < max_steps:
            p.stepSimulation(self.render_env.client_id)

            if video_writer is not None and curr_steps % 50 == 49:
                # if video_writer is not None:
                frame_width = 640
                frame_height = 480
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=frame_width,
                    height=frame_height,
                    viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 0, 0],
                        distance=5,
                        yaw=270,
                        pitch=-30,
                        roll=0,
                        upAxisIndex=2,
                    ),
                    projectionMatrix=p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=float(frame_width) / frame_height,
                        nearVal=0.1,
                        farVal=100.0,
                    ),
                )
                image = np.array(rgbImg, dtype=np.uint8)
                image = image[:, :, :3]

                # Add the frame to the video
                video_writer.append_data(image)

            curr_steps += 1
            if self.gui:
                time.sleep(1 / 240.0)
            if curr_steps % 1 == 0:
                contact = self.gripper.detect_contact(self.render_env.obj_id)

        # Give it another chance
        if contact:
            print("contact detected")

        return contact

    # def teleport(self, point, contact_vector, video_writer = None, standoff_d: float = 0.2):
    #     # Normalize contact vector.
    #     contact_vector = (contact_vector / contact_vector.norm(dim=-1)).float()

    #     p_teleport = torch.from_numpy(point).float()

    #     # breakpoint()

    #     e_z_init = torch.tensor([0, 0, 1.0]).float()
    #     e_y = -contact_vector
    #     e_x = torch.cross(-contact_vector, e_z_init)
    #     e_x = e_x / e_x.norm(dim=-1)
    #     e_z = torch.cross(e_x, e_y)
    #     e_z = e_z / e_z.norm(dim=-1)
    #     R_teleport = torch.stack([e_x, e_y, e_z], dim=1)
    #     R_gripper = torch.as_tensor(
    #         [
    #             [1, 0, 0],
    #             [0, 0, 1.0],
    #             [0, -1.0, 0],
    #         ]
    #     )
    #     # breakpoint()
    #     o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()

    #     self.gripper.set_pose(p_teleport, o_teleport)
    #     self.gripper.set_velocity([0, 0, 0], [0, 0, 0])
    #     p.stepSimulation(self.render_env.client_id)
    #     contact = self.gripper.detect_contact(self.render_env.obj_id)

    #     if not contact:
    #         self.gripper.set_pose(p_teleport, o_teleport)
    #         candidate_points = p.getClosestPoints(bodyA=self.gripper.body_id, bodyB=self.render_env.obj_id, distance=0.1)
    #         print(len(candidate_points), " points.")
    #         for i in range(min(50, len(candidate_points))):
    #             new_attach_point = candidate_points[i][6]
    #             p_teleport = torch.from_numpy(np.array(new_attach_point)).float()
    #             # print(new_attach_point)
    #             self.gripper.set_pose(p_teleport, o_teleport)
    #             self.gripper.set_velocity([0, 0, 0], [0, 0, 0])
    #             p.stepSimulation(self.render_env.client_id)
    #             contact = self.gripper.detect_contact(self.render_env.obj_id)
    #             if contact:
    #                 break
    #     # max_steps = 500
    #     # curr_steps = 0
    #     # self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
    #     # while not contact and curr_steps < max_steps:
    #     #     p.stepSimulation(self.render_env.client_id)

    #     #     if video_writer is not None and curr_steps % 50 == 49:
    #     #     # if video_writer is not None:
    #     #         frame_width = 640
    #     #         frame_height = 480
    #     #         width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=frame_width, height=frame_height, viewMatrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0,0,0], distance=5, yaw=90, pitch=-30, roll=0, upAxisIndex=2), projectionMatrix=p.computeProjectionMatrixFOV(fov=60, aspect=float(frame_width)/frame_height, nearVal=0.1, farVal=100.0))
    #     #         image = np.array(rgbImg, dtype=np.uint8)
    #     #         image = image[:, :, :3]

    #     #         # Add the frame to the video
    #     #         video_writer.append_data(image)

    #     #     curr_steps += 1
    #     #     if self.gui:
    #     #         time.sleep(1 / 240.0)
    #     #     if curr_steps % 1 == 0:
    #     #         contact = self.gripper.detect_contact(self.render_env.obj_id)

    #     if contact:
    #         print("contact detected")

    #     return contact

    def teleport(
        self, points, contact_vectors, video_writer=None, standoff_d: float = 0.2
    ):
        for id, (point, contact_vector) in enumerate(zip(points, contact_vectors)):
            # Normalize contact vector.
            contact_vector = (contact_vector / contact_vector.norm(dim=-1)).float()
            p_teleport = (torch.from_numpy(point) + contact_vector * standoff_d).float()
            # breakpoint()

            e_z_init = torch.tensor([0, 0, 1.0]).float()
            e_y = -contact_vector
            e_x = torch.cross(-contact_vector, e_z_init)
            e_x = e_x / e_x.norm(dim=-1)
            e_z = torch.cross(e_x, e_y)
            e_z = e_z / e_z.norm(dim=-1)
            R_teleport = torch.stack([e_x, e_y, e_z], dim=1)
            R_gripper = torch.as_tensor(
                [
                    [1, 0, 0],
                    [0, 0, 1.0],
                    [0, -1.0, 0],
                ]
            )
            # breakpoint()
            o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()

            self.gripper.set_pose(p_teleport, o_teleport)

            contact = self.gripper.detect_contact(self.render_env.obj_id)
            max_steps = 500
            curr_steps = 0
            self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
            while not contact and curr_steps < max_steps:
                p.stepSimulation(self.render_env.client_id)
                # print(point, p.getBasePositionAndOrientation(self.gripper.body_id),p.getBasePositionAndOrientation(self.gripper.base_id))
                if video_writer is not None and curr_steps % 50 == 49:
                    # if video_writer is not None:
                    frame_width = 640
                    frame_height = 480
                    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                        width=frame_width,
                        height=frame_height,
                        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                            cameraTargetPosition=[0, 0, 0],
                            distance=5,
                            yaw=270,
                            pitch=-30,
                            roll=0,
                            upAxisIndex=2,
                        ),
                        projectionMatrix=p.computeProjectionMatrixFOV(
                            fov=60,
                            aspect=float(frame_width) / frame_height,
                            nearVal=0.1,
                            farVal=100.0,
                        ),
                    )
                    image = np.array(rgbImg, dtype=np.uint8)
                    image = image[:, :, :3]

                    # Add the frame to the video
                    video_writer.append_data(image)

                curr_steps += 1
                if self.gui:
                    time.sleep(1 / 240.0)
                if curr_steps % 1 == 0:
                    contact = self.gripper.detect_contact(self.render_env.obj_id)

            # Give it another chance
            if contact:
                print("contact detected")
                return id, True

        return -1, False

    def attach(self):
        self.gripper.activate(self.render_env.obj_id)

    def pull(self, direction, n_steps: int = 100):
        direction = torch.as_tensor(direction)
        direction = direction / direction.norm(dim=-1)
        # breakpoint()
        for _ in range(n_steps):
            self.gripper.set_velocity(direction * 0.4, [0, 0, 0])
            p.stepSimulation(self.render_env.client_id)
            if self.gui:
                time.sleep(1 / 240.0)

    def get_joint_value(self, target_link: str):
        link_index = self.render_env.link_name_to_index[target_link]
        state = p.getJointState(
            self.render_env.obj_id, link_index, self.render_env.client_id
        )
        joint_pos = state[0]
        return joint_pos

    def detect_success(self, target_link: str):
        link_index = self.render_env.link_name_to_index[target_link]
        info = p.getJointInfo(
            self.render_env.obj_id, link_index, self.render_env.client_id
        )
        lower, upper = info[8], info[9]
        curr_pos = self.get_joint_value(target_link)

        sign = -1 if upper < lower else 1
        print(
            f"lower: {lower}, upper: {upper}, curr: {curr_pos}, success:{(upper - curr_pos) / (upper - lower) < 0.1}"
        )

        return (upper - curr_pos) / (upper - lower) < 0.1, (curr_pos - lower) / (
            upper - lower
        )

    def randomize_joints(self):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def randomize_specific_joints(self, joint_list):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def articulate_specific_joints(self, joint_list, amount):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = amount * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def randomize_joints_openclose(self, joint_list):
        randind = np.random.choice([0, 1])
        # Close: 0
        # Open: 1
        self.close_or_open = randind
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angles = [lower, upper]
                angle = angles[randind]
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )


@dataclass
class TrialResult:
    success: bool
    contact: bool
    assertion: bool
    init_angle: float
    final_angle: float
    now_angle: float

    # UMPNet metric goes here
    metric: float


class GTFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        links = raw_data.semantics.by_type("slider")
        links += raw_data.semantics.by_type("hinge")
        current_jas = {}
        for link in links:
            linkname = link.name
            chain = raw_data.obj.get_chain(linkname)
            for joint in chain:
                current_jas[joint.name] = 0

        normalized_flow = compute_normalized_flow(
            P_world,
            env.render_env.T_world_base,
            current_jas,
            pc_seg,
            env.render_env.link_name_to_index,
            raw_data,
            "all",
        )

        return torch.from_numpy(normalized_flow)

    def get_movable_mask(self, obs) -> torch.Tensor:
        flow = self(obs)
        mask = (~(np.isclose(flow, 0.0)).all(axis=-1)).astype(np.bool_)
        return mask


class GTTrajectoryModel:
    def __init__(self, raw_data, env, traj_len=20):
        self.raw_data = raw_data
        self.env = env
        self.traj_len = traj_len

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        links = raw_data.semantics.by_type("slider")
        links += raw_data.semantics.by_type("hinge")
        current_jas = {}
        for link in links:
            linkname = link.name
            chain = raw_data.obj.get_chain(linkname)
            for joint in chain:
                current_jas[joint.name] = 0
        trajectory, _ = compute_flow_trajectory(
            self.traj_len,
            P_world,
            env.render_env.T_world_base,
            current_jas,
            pc_seg,
            env.render_env.link_name_to_index,
            raw_data,
            "all",
        )
        return torch.from_numpy(trajectory)


def run_trial(
    env: PMSuctionSim,
    raw_data: PMObject,
    target_link: str,
    model,
    gt_model=None,  # When we use mask_input_channel=True, this is the mask generator
    n_steps: int = 30,
    n_pts: int = 1200,
    save_name: str = "unknown",
    website: bool = False,
    gui: bool = False,
) -> TrialResult:
    torch.manual_seed(42)
    sim_trajectory = [0.05] + [0] * (n_steps)  # start from 0.05

    if website:
        # Flow animation
        animation = FlowNetAnimation()

    # First, reset the environment.
    env.reset()
    # Joint information
    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    # Sometimes doors collide with themselves. It's dumb.
    if (
        raw_data.category == "Door"
        and raw_data.semantics.by_name(target_link).type == "hinge"
    ):
        env.set_joint_state(
            target_link, init_angle + 0.05 * (target_angle - init_angle)
        )
        # env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        env.set_joint_state(
            target_link, init_angle + 0.05 * (target_angle - init_angle)
        )
        # env.set_joint_state(target_link, 0.05)

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

    if init_angle == target_angle:  # Not movable
        p.disconnect(physicsClientId=env.render_env.client_id)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    # breakpoint()
    if gt_model is None:  # GT Flow model
        pred_trajectory = model(copy.deepcopy(pc_obs))
    else:
        movable_mask = gt_model.get_movable_mask(pc_obs)
        pred_trajectory = model(copy.deepcopy(pc_obs), movable_mask)
    # pred_trajectory = model(copy.deepcopy(pc_obs))
    # breakpoint()
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    traj_len = pred_trajectory.shape[1]  # Trajectory length
    print(f"Predicting {traj_len} length trajectories.")
    pred_flow = pred_trajectory[:, 0, :]

    # flow_fig(torch.from_numpy(P_world), pred_flow, sizeref=0.1, use_v2=True).show()
    # breakpoint()

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
    # assert link_ixs.any()
    if not link_ixs.any():
        p.disconnect(physicsClientId=env.render_env.client_id)
        print("link_ixs finds no point")
        animation_results = animation.animate() if website else None
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    if website:
        if gui:
            # Record simulation video
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                f"./logs/simu_eval/video_assets/{save_name}.mp4",
            )
        else:
            video_file = f"./logs/simu_eval/video_assets/{save_name}.mp4"
            # # cv2 output videos won't show on website
            frame_width = 640
            frame_height = 480
            # fps = 5
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
            # videoWriter.write(rgbImgOpenCV)

            # Camera param
            writer = imageio.get_writer(video_file, fps=5)

            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=frame_width,
                height=frame_height,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=5,
                    yaw=270,
                    pitch=-30,
                    roll=0,
                    upAxisIndex=2,
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=float(frame_width) / frame_height,
                    nearVal=0.1,
                    farVal=100.0,
                ),
            )
            image = np.array(rgbImg, dtype=np.uint8)
            image = image[:, :, :3]

            # Add the frame to the video
            writer.append_data(image)

    # The attachment point is the point with the highest flow.
    # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
    top_k_point = min(20, len(pred_flow[link_ixs]))
    best_flow_ix = torch.topk(pred_flow[link_ixs].norm(dim=-1), top_k_point)[1]
    if top_k_point == 1:
        best_flow_ix = torch.tensor(list(best_flow_ix) * 2)
    # print(best_flow_ix)
    # try:
    #     best_flow_ix = torch.topk(pred_flow[link_ixs].norm(dim=-1), int(len(pred_flow[link_ixs].norm(dim=-1)) / 2))[1][-1]  # Not on the edge
    # except:
    #     best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
    best_flow = pred_flow[link_ixs][best_flow_ix]
    best_point = P_world[link_ixs][best_flow_ix]
    # breakpoint()

    # Teleport to an approach pose, approach, the object and grasp.
    if website and not gui:
        # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
        best_flow_ix, contact = env.teleport(best_point, best_flow, video_writer=writer)
    else:
        # contact = env.teleport_and_approach(best_point, best_flow)
        best_flow_ix, contact = env.teleport(best_point, best_flow)
    best_flow = pred_flow[link_ixs][best_flow_ix]
    best_point = P_world[link_ixs][best_flow_ix]
    last_step_grasp_point = best_point

    if not contact:
        if website:
            segmented_flow = np.zeros_like(pred_flow)
            segmented_flow[link_ixs] = pred_flow[link_ixs]
            segmented_flow = np.array(
                normalize_trajectory(
                    torch.from_numpy(np.expand_dims(segmented_flow, 1))
                ).squeeze()
            )
            animation.add_trace(
                torch.as_tensor(P_world),
                torch.as_tensor([P_world]),
                torch.as_tensor([segmented_flow]),
                "red",
            )
            if gui:
                p.stopStateLogging(log_id)
            else:
                # Write video
                writer.close()
                # videoWriter.release()

        print("No contact!")
        p.disconnect(physicsClientId=env.render_env.client_id)
        animation_results = None if not website else animation.animate()
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=True,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    env.attach()

    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    success = False

    global_step = 0
    # for i in range(n_steps):
    while global_step < n_steps:
        # Predict the flow on the observation.
        if gt_model is None:  # GT Flow model
            pred_trajectory = model(copy.deepcopy(pc_obs))
        else:
            movable_mask = gt_model.get_movable_mask(pc_obs)
            # breakpoint()
            pred_trajectory = model(pc_obs, movable_mask)
            # pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(pred_trajectory.shape[1]):
            if global_step == n_steps:
                break
            global_step += 1
            pred_flow = pred_trajectory[:, traj_step, :]
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

            # Filter down just the points on the target link.
            # breakpoint()
            link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
            # assert link_ixs.any()
            if not link_ixs.any():
                if website:
                    if gui:
                        p.stopStateLogging(log_id)
                    else:
                        writer.close()
                        # videoWriter.release()
                p.disconnect(physicsClientId=env.render_env.client_id)
                print("link_ixs finds no point")
                animation_results = animation.animate() if website else None
                return (
                    animation_results,
                    TrialResult(
                        assertion=False,
                        success=False,
                        contact=False,
                        init_angle=0,
                        final_angle=0,
                        now_angle=0,
                        metric=0,
                    ),
                    sim_trajectory,
                )

            # Get the best direction.
            # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
            top_k_point = min(20, len(pred_flow[link_ixs]))
            best_flow_ix = torch.topk(pred_flow[link_ixs].norm(dim=-1), top_k_point)[1]
            if top_k_point == 1:
                best_flow_ix = torch.tensor(list(best_flow_ix) * 2)
            # print(best_flow_ix)
            best_flow = pred_flow[link_ixs][best_flow_ix]

            # (1) Strategy 1 - Don't change grasp point
            # (2) Strategy 2 - Change grasp point when leverage difference is large
            lev_diff_thres = 0.2
            move_dist_thres = 0.01
            # Only change if the new point's leverage is a great increase
            gripper_tip_pos = p.getClosestPoints(
                env.gripper.body_id, env.render_env.obj_id, distance=0.5, linkIndexA=0
            )[0][5]
            # gripper_tip_pos, _ = p.getBasePositionAndOrientation(env.gripper.body_id)
            pcd_dist = torch.tensor(P_world[link_ixs] - np.array(gripper_tip_pos)).norm(
                dim=-1
            )
            grasp_point_id = pcd_dist.argmin()
            lev_diff = best_flow.norm(dim=-1) - pred_flow[link_ixs][
                grasp_point_id
            ].norm(dim=-1)
            if (
                pcd_dist[grasp_point_id] < move_dist_thres
                or lev_diff[0] > lev_diff_thres
            ):  # pcd_dist < 0.05 -> didn't move much....
                env.reset_gripper()
                p.stepSimulation(
                    env.render_env.client_id
                )  # Make sure the constraint is lifted

                best_point = P_world[link_ixs][best_flow_ix]
                if website and not gui:
                    # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
                    best_flow_ix, contact = env.teleport(
                        best_point, best_flow, video_writer=writer
                    )
                else:
                    # contact = env.teleport_and_approach(best_point, best_flow)
                    best_flow_ix, contact = env.teleport(best_point, best_flow)
                best_flow = pred_flow[link_ixs][best_flow_ix]
                best_point = P_world[link_ixs][best_flow_ix]
                last_step_grasp_point = best_point  # Grasp a new point

                if not contact:
                    if website:
                        segmented_flow = np.zeros_like(pred_flow)
                        segmented_flow[link_ixs] = pred_flow[link_ixs]
                        segmented_flow = np.array(
                            normalize_trajectory(
                                torch.from_numpy(np.expand_dims(segmented_flow, 1))
                            ).squeeze()
                        )
                        animation.add_trace(
                            torch.as_tensor(P_world),
                            torch.as_tensor([P_world]),
                            torch.as_tensor([segmented_flow]),
                            "red",
                        )
                        if gui:
                            p.stopStateLogging(log_id)
                        else:
                            # Write video
                            writer.close()
                            # videoWriter.release()

                    print("No contact!")
                    p.disconnect(physicsClientId=env.render_env.client_id)
                    animation_results = None if not website else animation.animate()
                    return (
                        animation_results,
                        TrialResult(
                            success=False,
                            assertion=True,
                            contact=False,
                            init_angle=0,
                            final_angle=0,
                            now_angle=0,
                            metric=0,
                        ),
                        sim_trajectory,
                    )

                env.attach()
            else:
                best_flow = pred_flow[link_ixs][best_flow_ix[0]]
                last_step_grasp_point = P_world[link_ixs][
                    grasp_point_id
                ]  # The original point - don't need to change

            # # (3) Strategy 3 - Always change grasp point
            # # Only change if the new point's leverage is a great increase
            # env.reset_gripper()
            # best_point = P_world[link_ixs][best_flow_ix]
            # if website and not gui:
            #     # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
            #     best_flow_ix, contact = env.teleport(best_point, best_flow, video_writer=writer)
            # else:
            #     # contact = env.teleport_and_approach(best_point, best_flow)
            #     best_flow_ix, contact = env.teleport(best_point, best_flow)
            # best_flow = pred_flow[link_ixs][best_flow_ix]
            # best_point = P_world[link_ixs][best_flow_ix]
            # if not contact:
            #     if website:
            #         segmented_flow = np.zeros_like(pred_flow)
            #         segmented_flow[link_ixs] = pred_flow[link_ixs]
            #         segmented_flow = np.array(
            #             normalize_trajectory(
            #                 torch.from_numpy(np.expand_dims(segmented_flow, 1))
            #             ).squeeze()
            #         )
            #         animation.add_trace(
            #             torch.as_tensor(P_world),
            #             torch.as_tensor([P_world]),
            #             torch.as_tensor([segmented_flow]),
            #             "red",
            #         )
            #         if gui:
            #             p.stopStateLogging(log_id)
            #         else:
            #             # Write video
            #             writer.close()
            #             # videoWriter.release()

            #     print("No contact!")
            #     p.disconnect(physicsClientId=env.render_env.client_id)
            #     animation_results = None if not website else animation.animate()
            #     return animation_results, TrialResult(
            #         success=False,
            #         assertion=True,
            #         contact=False,
            #         init_angle=0,
            #         final_angle=0,
            #         now_angle=0,
            #         metric=0,
            #     ), sim_trajectory

            env.attach()
            # Perform the pulling.
            # if best_flow.sum() == 0:
            #     continue
            # print(best_flow)
            env.pull(best_flow)
            env.attach()

            if website:
                # Add pcd to flow animation
                segmented_flow = np.zeros_like(pred_flow)
                segmented_flow[link_ixs] = pred_flow[link_ixs]
                segmented_flow = np.array(
                    normalize_trajectory(
                        torch.from_numpy(np.expand_dims(segmented_flow, 1))
                    ).squeeze()
                )
                animation.add_trace(
                    torch.as_tensor(P_world),
                    torch.as_tensor([P_world]),
                    torch.as_tensor([segmented_flow]),
                    "red",
                )

                # Capture frame
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=frame_width,
                    height=frame_height,
                    viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 0, 0],
                        distance=5,
                        yaw=270,
                        pitch=-30,
                        roll=0,
                        upAxisIndex=2,
                    ),
                    projectionMatrix=p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=float(frame_width) / frame_height,
                        nearVal=0.1,
                        farVal=100.0,
                    ),
                )
                # rgbImgOpenCV = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
                # videoWriter.write(rgbImgOpenCV)
                image = np.array(rgbImg, dtype=np.uint8)
                image = image[:, :, :3]

                # Add the frame to the video
                writer.append_data(image)

            success, sim_trajectory[global_step] = env.detect_success(target_link)

            if success:
                for left_step in range(global_step, 31):
                    sim_trajectory[left_step] = sim_trajectory[global_step]
                break

            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)

        if success:
            for left_step in range(global_step, 31):
                sim_trajectory[left_step] = sim_trajectory[global_step]
            break

    # calculate the metrics
    curr_pos = env.get_joint_value(target_link)
    metric = (curr_pos - init_angle) / (target_angle - init_angle)
    metric = min(max(metric, 0), 1)

    if website:
        if gui:
            p.stopStateLogging(log_id)
        else:
            writer.close()
            # videoWriter.release()

    p.disconnect(physicsClientId=env.render_env.client_id)
    animation_results = None if not website else animation.animate()
    return (
        animation_results,
        TrialResult(  # Save the flow visuals
            success=success,
            contact=True,
            assertion=True,
            init_angle=init_angle,
            final_angle=target_angle,
            now_angle=curr_pos,
            metric=metric,
        ),
        sim_trajectory,
    )
