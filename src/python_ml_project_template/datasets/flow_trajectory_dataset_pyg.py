from typing import List, Optional, Protocol, Union, cast

import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd

from python_ml_project_template.datasets.flow_trajectory_dataset import (
    FlowTrajectoryDataset,
)


class FlowTrajectoryTGData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor  # Points in the point cloud.
    delta: torch.Tensor  # instantaneous positive 3D flow trajectories.
    point: torch.Tensor  # the trajectory waypoints
    mask: torch.Tensor  # Mask of the part of interest.


class FlowTrajectoryPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        trajectory_len: int = 5,
        n_points: Optional[int] = 1200,
        seed: int = 42,  # Randomize everything
    ) -> None:
        super().__init__()
        self.dataset = FlowTrajectoryDataset(
            root,
            split,
            randomize_joints,
            randomize_camera,
            trajectory_len,
            n_points,
        )
        self.seed = seed

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self.dataset._dataset._ids[index], seed=self.seed)

    @staticmethod
    def get_processed_dir(randomize_joints, randomize_camera, trajectory_len):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        return f"processed_{trajectory_len}_{joint_chunk}_{camera_chunk}"

    def get_data(self, obj_id: str, seed) -> FlowTrajectoryTGData:
        data_dict = self.dataset.get_data(obj_id, seed)
        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            delta=torch.from_numpy(data_dict["delta"]).float(),
            point=torch.from_numpy(data_dict["point"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
        return cast(FlowTrajectoryTGData, data)
