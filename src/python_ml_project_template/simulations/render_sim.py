import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pybullet as p
import pybullet_data

from python_ml_project_template.simulations.camera import Camera
from python_ml_project_template.simulations.utils import (
    get_obj_z_offset,
    isnotebook,
    suppress_stdout,
)

# from part_embedding.flow_prediction.cam_utils import sample_az_ele


class PMRenderEnv:
    def __init__(
        self,
        obj_id: str,
        dataset_path: str,
        camera_pos: List = [-2, 0, 2],
        gui: bool = False,
        with_plane: bool = True,
    ):
        self.with_plane = with_plane
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)

        # Add in a plane.
        if with_plane:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        # Add in gravity.
        p.setGravity(0, 0, 0, self.client_id)

        # Add in the object.
        self.obj_id_str = obj_id
        obj_urdf = os.path.join(dataset_path, obj_id, "mobility.urdf")

        if isnotebook():
            self.obj_id = p.loadURDF(
                obj_urdf,
                useFixedBase=True,
                # flags=p.URDF_MAINTAIN_LINK_ORDER,
                physicsClientId=self.client_id,
            )

        else:
            with suppress_stdout():
                self.obj_id = p.loadURDF(
                    obj_urdf,
                    useFixedBase=True,
                    # flags=p.URDF_MAINTAIN_LINK_ORDER,
                    physicsClientId=self.client_id,
                )

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

        # Create a camera.
        self.camera = Camera(pos=camera_pos, znear=0.01, zfar=10)

        # From https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=12728&p=42293&hilit=linkIndex#p42293
        self.link_name_to_index = {
            p.getBodyInfo(self.obj_id, physicsClientId=self.client_id)[0].decode(
                "UTF-8"
            ): -1,
        }

        # Get the segmentation.
        for _id in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            _name = p.getJointInfo(self.obj_id, _id, physicsClientId=self.client_id)[
                12
            ].decode("UTF-8")
            self.link_name_to_index[_name] = _id

    def render(self, return_prgb=False, link_seg=True):
        if not return_prgb:
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb, self.with_plane, link_seg
            )
            return rgb, depth, seg, P_cam, P_world, pc_seg, segmap
        else:
            rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb, self.with_plane, link_seg
            )
            return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap

    def set_camera(self, camera_pos: List[int]):
        self.camera.set_camera_position(camera_pos)

    def randomize_joints(self, joints: Optional[Sequence[str]] = None):
        if joints is not None:
            joints_set = set(joints)
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)

            # Skip joint if we are filtering and it's not selected.
            if joints is not None and jinfo[1].decode("UTF-8") not in joints_set:
                continue

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

    def set_joint_angles(self, joints: Dict[str, float]):
        """Sets joint angles, interpolating between lower and upper."""
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            joint_name = jinfo[1].decode("UTF-8")
            # Skip joint if we are filtering and it's not selected.
            if joint_name not in joints:
                continue

            # Interpolate between upper and lower.
            assert 0.0 <= joints[joint_name] <= 1.0
            lower, upper = jinfo[8], jinfo[9]
            angle = joints[joint_name] * (upper - lower) + lower
            p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def get_joint_angles(self) -> Dict[str, float]:
        angles = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            jstate = p.getJointState(self.obj_id, i, self.client_id)
            angles[jinfo[1].decode("UTF-8")] = jstate[0]
        return angles

    # def randomize_camera(self):
    #     x, y, z, az, el = sample_az_ele(
    #         np.sqrt(8), np.deg2rad(30), np.deg2rad(150), np.deg2rad(30), np.deg2rad(60)
    #     )
    #     self.camera.set_camera_position((x, y, z))
