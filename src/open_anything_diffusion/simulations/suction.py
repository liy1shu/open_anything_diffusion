import copy
import functools
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pybullet as p
import pybullet_data
import torch

# from open_anything_diffusion.simulations.floating_vacuum_gripper import (
#     FloatingSuctionGripper,
# )
from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper
from scipy.spatial.transform import Rotation as R

from open_anything_diffusion.simulations.calc_art import compute_new_points
from open_anything_diffusion.simulations.camera import Camera
from open_anything_diffusion.simulations.pm_raw import PMRawData
from open_anything_diffusion.simulations.utils import get_obj_z_offset, suppress_stdout

# from python_ml_project_template.datasets.flow_trajectory_dataset import compute_normalized_flow, compute_flow_trajectory
# from part_embedding.flow_prediction.cam_utils import sample_az_ele

# import pkgutil
# egl = pkgutil.get_loader("eglRenderer")

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
# CAMERA_INTRINSICS = np.array(
#     [
#         [450, 0, RENDER_WIDTH / 2],
#         [0, 450, RENDER_HEIGHT / 2],
#         [0, 0, 1],
#     ]
# )
_scene_cam_fov_w = 69.40
_scene_cam_focal_length = (float(640) / 2) / np.tan(
    (np.pi * _scene_cam_fov_w / 180) / 2
)
CAMERA_INTRINSICS = np.array(
    [
        [
            _scene_cam_focal_length,
            0,
            640 / 2,
        ],
        [
            0,
            _scene_cam_focal_length,
            480 / 2,
        ],
        [0, 0, 1],
    ]
)


def fetch_mobility_object(category_name, instance_id):
    mobility_path = "/home/harry/umpnet/mobility_dataset"
    urdf_path = os.path.join(mobility_path, category_name, instance_id, "mobility.urdf")
    with open(
        os.path.join(
            mobility_path, category_name, instance_id, "object_meta_info.json"
        ),
        "r",
    ) as f:
        meta_info = json.load(f)
    orientation = meta_info["orientation"]
    offset = [0, 0, meta_info["offset_z"]]
    scale = meta_info["scale"]
    return (
        orientation,
        offset,
        scale,
        meta_info["moveable_link"],
    )


def is_overlapping(int1, int2):
    return int1[1] > int2[0] and int2[1] > int1[0]


def get_obj_bbox_xy(object_id, sim_id):
    bboxes = [p.getAABB(object_id, physicsClientId=sim_id)]  # Add the base one.
    for i in range(p.getNumJoints(object_id, physicsClientId=sim_id)):
        aabb = p.getAABB(object_id, i, physicsClientId=sim_id)  # Add the links.
        bboxes.append(aabb)
    # xmin = np.array(bboxes)[:, :, 0].min()
    xmin = functools.reduce(lambda a, b: min(a, b[0][0]), bboxes, 0.0)
    # xmax = np.array(bboxes)[:, :, 0].max()
    xmax = functools.reduce(lambda a, b: max(a, b[1][0]), bboxes, 0.0)
    # ymin = np.array(bboxes)[:, :, 1].min()
    ymin = functools.reduce(lambda a, b: min(a, b[0][1]), bboxes, 0.0)
    # ymax = np.array(bboxes)[:, :, 1].max()
    ymax = functools.reduce(lambda a, b: max(a, b[1][1]), bboxes, 0.0)
    return [[xmin, xmax], [ymin, ymax]]


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def fetch_mobility_object(category_name, instance_id):
    mobility_path = "/home/harry/umpnet/mobility_dataset"
    urdf_path = os.path.join(mobility_path, category_name, instance_id, "mobility.urdf")
    with open(
        os.path.join(
            mobility_path, category_name, instance_id, "object_meta_info.json"
        ),
        "r",
    ) as f:
        meta_info = json.load(f)
    orientation = meta_info["orientation"]
    offset = [0, 0, meta_info["offset_z"]]
    scale = meta_info["scale"]
    return (
        orientation,
        offset,
        scale,
        meta_info["moveable_link"],
    )


def is_overlapping(int1, int2):
    return int1[1] > int2[0] and int2[1] > int1[0]


def check_collision_placing(ranges, bbox, pos_x, pos_y):
    if not ranges:
        return False
    else:
        xbound = [
            pos_x - (bbox[0][1] - bbox[0][0]) / 2,
            pos_x + (bbox[0][1] - bbox[0][0]) / 2,
        ]
        ybound = [
            pos_y - (bbox[1][1] - bbox[1][0]) / 2,
            pos_y + (bbox[1][1] - bbox[1][0]) / 2,
        ]
        for obj in ranges:
            obj_xbound = ranges[obj][0]
            obj_ybound = ranges[obj][1]
            if is_overlapping(xbound, obj_xbound) and is_overlapping(
                ybound, obj_ybound
            ):
                return True

        return False


def compute_flow(
    P_world,
    T_world_base,
    pc_seg,
    pm_raw_data: PMRawData,
    linkname: str = "all",
    linkname_to_id={},
    close_open=None,
):
    flow = np.zeros_like(P_world)

    # joint_states = np.zeros(len(chain))
    def _compute_flow(link_name):
        # todo: convert from link_name to link id
        try:
            link_id = linkname_to_id[link_name]
        except:
            breakpoint()
        link_ixs = pc_seg == link_id
        filtered_pc = P_world[link_ixs]

        chain = pm_raw_data.obj.get_chain(link_name)
        current_ja = np.zeros(len(chain))
        target_ja = np.zeros(len(chain))
        target_ja[-1] = 0.01
        if close_open == 0:
            target_ja[-1] = 0.01
        elif close_open == 1:
            target_ja[-1] = -0.01

        filtered_new_pc = compute_new_points(
            filtered_pc, T_world_base, chain, current_ja=current_ja, target_ja=target_ja
        )

        part_flow = filtered_new_pc - filtered_pc
        if close_open != None:
            part_flow = -part_flow
        flow[link_ixs] = part_flow

    if linkname != "all":
        for l in linkname:
            _compute_flow(l)
    else:
        links = pm_raw_data.semantics.by_type("slider")
        links += pm_raw_data.semantics.by_type("hinge")

        for link in links:
            _compute_flow(link.name)

    return flow


class PMSuctionSimMultiObj:
    def __init__(self, obj_ids: list, dataset_path: str, gui: bool = False):
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)

        # Add in a plane.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Add in gravity.
        p.setGravity(0, 0, 0, self.client_id)

        # Add in the objects.
        self.obj_ids_str = obj_ids
        # Object IDs in Pybullet are expressed as a dictionary where the keys are PMDataset ID and the vals are body ids.
        self.obj_ids = {}
        for i in self.obj_ids_str:
            obj_urdf = os.path.join(dataset_path, i, "mobility.urdf")
            self.obj_ids[i] = p.loadURDF(
                obj_urdf,
                useFixedBase=True,
                physicsClientId=self.client_id,
            )

        # The object isn't placed at the bottom of the scene.
        self.T_world_base: Dict = {}
        self.minzs: Dict = {}

        # Randomize placement
        self.xy_ranges: Dict = {}

        self.add_objects()

        # todo: Create a Suction Gripper that can detect multiple contacts. i.e. obj_id is dynamic.
        # self.gripper = FloatingSuctionGripper(
        #     list(self.obj_ids.keys())[0], self.client_id
        # )
        # self.gripper.set_pose(
        #     [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        # )

        self.camera = Camera(pos=[-4, 0, 4])

        # From https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=12728&p=42293&hilit=linkIndex#p42293
        self.link_name_to_index_group = {}
        for _, obj_id in self.obj_ids.items():
            self.link_name_to_index_group[obj_id] = {
                p.getBodyInfo(obj_id, physicsClientId=self.client_id)[0].decode(
                    "UTF-8"
                ): -1,
            }

            for _id in range(p.getNumJoints(obj_id, physicsClientId=self.client_id)):
                _name = p.getJointInfo(obj_id, _id, physicsClientId=self.client_id)[
                    12
                ].decode("UTF-8")
                self.link_name_to_index_group[obj_id][_name] = _id

    def get_category(self, obj_id):
        split_file = json.load(
            open("/home/harry/umpnet/mobility_dataset/split-full.json")
        )
        for cat in split_file["train"]:
            if obj_id in split_file["train"][cat]["train"]:
                return cat
            elif obj_id in split_file["train"][cat]["test"]:
                return cat
        for cat in split_file["test"]:
            if obj_id in split_file["test"][cat]["test"]:
                return cat
        raise NotImplementedError

    def add_objects(self):
        for instance_id, obj_id in self.obj_ids.items():
            # Some meta data
            category_name = self.get_category(instance_id)
            orientation, offset, scaling, moveable_link_dict = fetch_mobility_object(
                category_name, instance_id
            )
            self.minzs[obj_id] = get_obj_z_offset(obj_id, self.client_id)

            pos_x = np.random.uniform(-0.3, 0.3)
            pos_y = np.random.uniform(-3, 3)
            bbox = get_obj_bbox_xy(obj_id, self.client_id)
            while check_collision_placing(self.xy_ranges, bbox, pos_x, pos_y):
                pos_x = np.random.uniform(-2, 2)
                pos_y = np.random.uniform(-0.2, 0.2)
            xbound = [
                pos_x - (bbox[0][1] - bbox[0][0]) / 2,
                pos_x + (bbox[0][1] - bbox[0][0]) / 2,
            ]
            ybound = [
                pos_y - (bbox[1][1] - bbox[1][0]) / 2,
                pos_y + (bbox[1][1] - bbox[1][0]) / 2,
            ]
            self.xy_ranges[obj_id] = [xbound, ybound]

            p.resetBasePositionAndOrientation(
                obj_id,
                posObj=[pos_x, pos_y, -self.minzs[obj_id]],
                ornObj=[0, 0, 0, 1],
                physicsClientId=self.client_id,
            )
            self.T_world_base[obj_id] = np.eye(4)
            self.T_world_base[obj_id][:3, -1] = [0, 0, -self.minzs[obj_id]]

    def reset(self):
        pass

    def set_gripper_pose(self):
        pass

    def set_object_state(self):
        pass

    def render(self, return_prgb=False):
        if not return_prgb:
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb
            )
            return rgb, depth, seg, P_cam, P_world, pc_seg, segmap
        else:
            rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb
            )
            return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap

    def set_camera(self):
        pass

    def teleport_and_approach(self, point, contact_vector):
        pass

    def attach(self):
        self.gripper.activate()

    def pull(self, direction):
        pass

    def detect_success(self):
        pass

    def randomize_joints(self):
        for _, obj_id in self.obj_ids.items():
            for i in range(p.getNumJoints(obj_id, self.client_id)):
                jinfo = p.getJointInfo(obj_id, i, self.client_id)
                if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
                    lower, upper = jinfo[8], jinfo[9]
                    angle = np.random.random() * (upper - lower) + lower
                    p.resetJointState(obj_id, i, angle, 0, self.client_id)

    # def randomize_camera(self):
    #     x, y, z, az, el = sample_az_ele(
    #         np.sqrt(8), np.deg2rad(30), np.deg2rad(150), np.deg2rad(30), np.deg2rad(60)
    #     )
    #     self.camera.set_camera_position((x, y, z))


class PMSuctionSim:
    def __init__(self, obj_id: str, dataset_path: str, gui: bool = False):
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        self.gui = gui
        # Add in a plane.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Add in gravity.
        p.setGravity(0, 0, 0, self.client_id)

        # Add in the object.
        self.obj_id_str = obj_id
        obj_urdf = os.path.join(dataset_path, obj_id, "mobility.urdf")
        with suppress_stdout():
            self.obj_id = p.loadURDF(
                obj_urdf,
                useFixedBase=True,
                # flags=p.URDF_MAINTAIN_LINK_ORDER,
                physicsClientId=self.client_id,
            )

        # plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin", self.client_id)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
        # )
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

        # The object isn't placed at the bottom of the scene.
        self.minz = get_obj_z_offset(self.obj_id, self.client_id)
        p.resetBasePositionAndOrientation(
            self.obj_id,
            posObj=[0, 0, -self.minz],
            ornObj=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        self.T_world_base = np.eye(4)
        self.T_world_base[2, 3] = -self.minz

        # Add in the robot.
        pos, orient = [-1, 0, 1], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        # self.gripper = FloatingSuctionGripper(self.client_id, self.obj_id)
        self.gripper = FloatingSuctionGripper(self.client_id)
        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )
        self.gripper.activate(self.obj_id)

        self.camera = Camera(pos=[-3, 0, 1.2], znear=0.01, zfar=10)

        # From https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=12728&p=42293&hilit=linkIndex#p42293
        self.link_name_to_index = {
            p.getBodyInfo(self.obj_id, physicsClientId=self.client_id)[0].decode(
                "UTF-8"
            ): -1,
        }

        for _id in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            _name = p.getJointInfo(self.obj_id, _id, physicsClientId=self.client_id)[
                12
            ].decode("UTF-8")
            self.link_name_to_index[_name] = _id

    def run_demo(self):
        while True:
            self.gripper.set_velocity([0.4, 0, 0.0], [0, 0, 0])
            for i in range(10):
                p.stepSimulation(self.client_id)
                time.sleep(1 / 240.0)
            contact = self.gripper.detect_contact()
            if contact:
                break

        print("stopping gripper")

        self.gripper.set_velocity([0.001, 0, 0.0], [0, 0, 0])
        for i in range(10):
            p.stepSimulation(self.client_id)
            time.sleep(1 / 240.0)
            contact = self.gripper.detect_contact()
            print(contact)

        print("starting activation")

        self.gripper.activate()

        self.gripper.set_velocity([0, 0, 0.0], [0, 0, 0])
        for i in range(100):
            p.stepSimulation(self.client_id)
            time.sleep(1 / 240.0)

        # print("releasing")
        # self.gripper.release()

        print("starting motion")
        for i in range(100):
            p.stepSimulation(self.client_id)
            time.sleep(1 / 240.0)

        for _ in range(20):
            for i in range(100):
                self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
                self.gripper.apply_force([-500, 0, 0])
                p.stepSimulation(self.client_id)
                time.sleep(1 / 240.0)

            for i in range(100):
                self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
                self.gripper.apply_force([-500, 0, 0])
                p.stepSimulation(self.client_id)
                time.sleep(1 / 240.0)

        print("releasing")
        self.gripper.release()

        for i in range(1000):
            p.stepSimulation(self.client_id)
            time.sleep(1 / 240.0)

    def reset(self):
        pass

    def set_gripper_pose(self, pos, ori):
        self.gripper.set_pose(pos, ori)

    def set_joint_state(self, link_name: str, value: float):
        p.resetJointState(
            self.obj_id, self.link_name_to_index[link_name], value, 0.0, self.client_id
        )

    # def render(self, return_prgb=False):
    #     if not return_prgb:
    #         rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.camera.render(
    #             self.client_id, return_prgb
    #         )
    #         return rgb, depth, seg, P_cam, P_world, pc_seg, segmap
    #     else:
    #         rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.camera.render(
    #             self.client_id, return_prgb
    #         )
    #         return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap
    def render(self, filter_nonobj_pts: bool = False, n_pts: Optional[int] = None):
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.camera.render(
            self.client_id
        )

        if filter_nonobj_pts:
            pc_seg_obj = np.ones_like(pc_seg) * -1
            for k, (body, link) in segmap.items():
                if body == self.obj_id:
                    ixs = pc_seg == k
                    pc_seg_obj[ixs] = link

            is_obj = pc_seg_obj != -1
            P_cam = P_cam[is_obj]
            P_world = P_world[is_obj]
            # pc_seg = pc_seg[is_obj]
            pc_seg = pc_seg_obj[is_obj]
            # breakpoint()

        if n_pts is not None:
            perm = np.random.permutation(len(P_world))[:n_pts]
            P_cam = P_cam[perm]
            P_world = P_world[perm]
            pc_seg = pc_seg[perm]

        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap

    def set_camera(self):
        pass

    def teleport_and_approach(self, point, contact_vector, standoff_d: float = 0.2):
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

        contact = self.gripper.detect_contact()
        max_steps = 500
        curr_steps = 0
        self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
        while not contact and curr_steps < max_steps:
            p.stepSimulation(self.client_id)
            curr_steps += 1
            if self.gui:
                time.sleep(1 / 240.0)
            if curr_steps % 10 == 0:
                contact = self.gripper.detect_contact()

        if contact:
            print("contact detected")

        return contact

    def attach(self):
        self.gripper.activate()

    def pull(self, direction, n_steps: int = 100):
        direction = torch.as_tensor(direction)
        direction = direction / direction.norm(dim=-1)
        # breakpoint()
        for _ in range(n_steps):
            self.gripper.set_velocity(direction * 0.4, [0, 0, 0])
            p.stepSimulation(self.client_id)
            if self.gui:
                time.sleep(1 / 240.0)

    def get_joint_value(self, target_link: str):
        link_index = self.link_name_to_index[target_link]
        state = p.getJointState(self.obj_id, link_index, self.client_id)
        joint_pos = state[0]
        return joint_pos

    def detect_success(self, target_link: str):
        link_index = self.link_name_to_index[target_link]
        info = p.getJointInfo(self.obj_id, link_index, self.client_id)
        lower, upper = info[8], info[9]
        curr_pos = self.get_joint_value(target_link)

        print(f"lower: {lower}, upper: {upper}, curr: {curr_pos}")

        sign = -1 if upper < 0 else 1
        return sign * (upper - curr_pos) < 0.001

    def randomize_joints(self):
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def randomize_specific_joints(self, joint_list):
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def articulate_specific_joints(self, joint_list, amount):
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = amount * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def randomize_joints_openclose(self, joint_list):
        randind = np.random.choice([0, 1])
        # Close: 0
        # Open: 1
        self.close_or_open = randind
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angles = [lower, upper]
                angle = angles[randind]
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    # def randomize_camera(self):
    #     x, y, z, az, el = sample_az_ele(
    #         np.sqrt(8), np.deg2rad(30), np.deg2rad(150), np.deg2rad(30), np.deg2rad(60)
    #     )
    #     self.camera.set_camera_position((x, y, z))


@dataclass
class TrialResult:
    success: bool
    init_angle: float
    final_angle: float

    # UMPNet metric goes here
    metric: float


# TODO: change to the existing flow calculation method (No need to keep two functions)
class GTFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        flow = compute_flow(
            P_world,
            env.T_world_base,
            pc_seg,
            raw_data,
            linkname="all",  # lys
            linkname_to_id=env.link_name_to_index,
        )
        nonzero_flow_xs = (flow != 0.0).any(axis=-1)

        if nonzero_flow_xs.sum() > 0.0:
            nonzero_flow = flow[(flow != 0.0).any(axis=-1)]
            # print(f"there's zero flow for {env.obj_id_str}")
            # raise ValueError(f"there's zero flow for {self.obj_id_str}")
            largest = np.linalg.norm(nonzero_flow, axis=-1).max()

            flow = flow / largest

        return torch.from_numpy(flow)


class GTTrajectoryModel:
    # TODO: Generates trajectory
    def __init__(self, raw_data, env, traj_len=20):
        self.raw_data = raw_data
        self.env = env
        self.traj_len = traj_len

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        trajectory = np.zeros((P_world.shape[0], self.traj_len, 3))

        for step in range(self.traj_len):
            flow = compute_flow(
                P_world,
                env.T_world_base,
                pc_seg,
                raw_data,
                linkname="all",
                linkname_to_id=env.link_name_to_index,
            )
            nonzero_flow_xs = (flow != 0.0).any(axis=-1)

            P_world += flow

            if nonzero_flow_xs.sum() > 0.0:
                nonzero_flow = flow[(flow != 0.0).any(axis=-1)]
                # print(f"there's zero flow for {env.obj_id_str}")
                # raise ValueError(f"there's zero flow for {self.obj_id_str}")
                largest = np.linalg.norm(nonzero_flow, axis=-1).max()

                flow = flow / largest

            trajectory[:, step, :] = flow

        return torch.from_numpy(trajectory)


def run_trial(
    env: PMSuctionSim,
    raw_data: PMRawData,
    target_link: str,
    model,
    n_steps: int = 30,
    n_pts: int = 1200,
    traj_len: int = 1,  # By default, only move one step
) -> TrialResult:
    # First, reset the environment.
    env.reset()

    # Sometimes doors collide with themselves. It's dumb.
    # if raw_data.category == "Door" and raw_data.semantics.by_name(target_link).type == "hinge":
    #     env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        env.set_joint_state(target_link, 0.05)

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

    pred_trajectory = model(copy.deepcopy(pc_obs))
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    pred_flow = pred_trajectory[:, 0, :]

    # flow_fig(torch.from_numpy(P_world), pred_flow, sizeref=0.1, use_v2=True).show()
    # breakpoint()

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env.link_name_to_index[target_link]
    assert link_ixs.any()

    # The attachment point is the point with the highest flow.
    best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
    best_flow = pred_flow[link_ixs][best_flow_ix]
    best_point = P_world[link_ixs][best_flow_ix]

    # Teleport to an approach pose, approach, the object and grasp.
    contact = env.teleport_and_approach(best_point, best_flow)

    if not contact:
        return TrialResult(
            success=False,
            init_angle=0,
            final_angle=0,
            metric=0,
        )

    env.attach()

    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    success = False

    for i in range(n_steps):
        # Predict the flow on the observation.
        pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(traj_len):
            pred_flow = pred_trajectory[:, traj_step, :]
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

            # Filter down just the points on the target link.
            link_ixs = pc_seg == env.link_name_to_index[target_link]
            assert link_ixs.any()

            # Get the best direction.
            best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
            best_flow = pred_flow[link_ixs][best_flow_ix]

            # Perform the pulling.
            env.pull(best_flow)

            success = env.detect_success(target_link)

            if success:
                break

            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)

    init_angle = 0.0
    final_angle = 0.0
    metric = 0.0

    return TrialResult(
        success=success,
        init_angle=init_angle,
        final_angle=final_angle,
        metric=metric,
    )
