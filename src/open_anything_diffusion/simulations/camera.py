from typing import Dict, Optional, Tuple

import numpy as np
import pybullet as p

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
CAMERA_INTRINSICS = np.array(
    [
        [450, 0, RENDER_WIDTH / 2],
        [0, 450, RENDER_HEIGHT / 2],
        [0, 0, 1],
    ]
)

T_CAMGL_2_CAM = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)


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


class Camera:
    def __init__(
        self,
        pos,
        render_height=RENDER_HEIGHT,
        render_width=RENDER_WIDTH,
        znear=0.01,
        zfar=6,
        intrinsics=CAMERA_INTRINSICS,
        target=None,
    ):
        #######################################
        # First, compute the projection matrix.
        #######################################
        self.intrinsics = intrinsics
        focal_length = intrinsics[0][0]
        self.znear, self.zfar = znear, zfar
        self.fovh = (np.arctan((render_height / 2) / focal_length) * 2 / np.pi) * 180
        self.render_width = render_width
        self.render_height = render_height

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = render_width / render_height
        self.proj_list = p.computeProjectionMatrixFOV(
            self.fovh, aspect_ratio, self.znear, self.zfar
        )

        #######################################
        # Next, compute the view matrix.
        #######################################
        if target is None:
            target = [0, 0, 0.5]
        self.target = target
        self.view_list = self.__view_list(pos, target)

    @property
    def view_list(self):
        return self._view_list

    @view_list.setter
    def view_list(self, value):
        self._view_list = value
        self.T_camgl2world = np.asarray(value).reshape(4, 4).T
        self.T_world2camgl = np.linalg.inv(self.T_camgl2world)
        self.T_world2cam = self.T_world2camgl @ T_CAMGL_2_CAM

    @staticmethod
    def __view_list(eye, target):
        up = [0.0, 0.0, 1.0]
        target = target
        view_list = p.computeViewMatrix(eye, target, up)
        return view_list

    def set_camera_position(self, pos):
        self.view_list = self.__view_list(pos, self.target)

    def render(
        self, client_id, return_prgb=False, has_plane=True, link_seg=True
    ) -> Tuple[np.ndarray, ...]:
        if link_seg:
            _, _, rgb, zbuffer, seg = p.getCameraImage(
                RENDER_WIDTH,
                RENDER_HEIGHT,
                self.view_list,
                self.proj_list,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=client_id,
            )
        else:
            _, _, rgb, zbuffer, seg = p.getCameraImage(
                RENDER_WIDTH,
                RENDER_HEIGHT,
                self.view_list,
                self.proj_list,
                physicsClientId=client_id,
            )

        # Sometimes on mac things get weird.
        if isinstance(rgb, tuple):
            rgb = np.asarray(rgb).reshape(RENDER_HEIGHT, RENDER_WIDTH, 4)
            zbuffer = np.asarray(zbuffer).reshape(RENDER_HEIGHT, RENDER_WIDTH)
            seg = np.asarray(seg).reshape(RENDER_HEIGHT, RENDER_WIDTH)

        zfar, znear = self.zfar, self.znear
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth

        P_cam = get_pointcloud(depth, self.intrinsics)
        foreground_ixs = seg > 0 if has_plane else seg > -1
        pc_seg = seg[foreground_ixs].flatten()
        P_cam = P_cam[foreground_ixs]
        P_cam = P_cam.reshape(-1, 3)
        P_rgb = rgb[foreground_ixs]
        P_rgb = P_rgb[:, :3].reshape(-1, 3)

        Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
        Ph_world = (self.T_world2cam @ Ph_cam.T).T
        P_world = Ph_world[:, :3]

        # Undoing the bitmask so we can get the obj_id, link_index
        segmap: Optional[Dict]
        if link_seg:
            segmap = {
                label: ((label & ((1 << 24) - 1)), (label >> 24) - 1)
                for label in np.unique(seg)
            }
        else:
            segmap = None

        if return_prgb:
            return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap  # type: ignore

        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap  # type: ignore
