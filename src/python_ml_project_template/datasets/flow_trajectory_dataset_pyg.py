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
    trajectory: torch.Tensor  # instantaneous positive 3D flow trajectories.
    mask: torch.Tensor  # Mask of the part of interest.


class FlowTrajectoryPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,
        randomize_camera: bool = True,
        trajectory_len: int = 5,
        mode: str = "delta",
        n_points: Optional[int] = 1200,
    ) -> None:
        super().__init__()
        self.dataset = FlowTrajectoryDataset(
            root,
            split,
            randomize_joints,
            randomize_camera,
            trajectory_len,
            mode,
            n_points,
        )

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self.dataset._dataset._ids[index], seed=None)

    @staticmethod
    def get_processed_dir(randomize_joints, randomize_camera, trajectory_len, mode):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        return f"processed_{mode}_{trajectory_len}_{joint_chunk}_{camera_chunk}"

    def get_data(self, obj_id: str, seed=None) -> FlowTrajectoryTGData:
        data_dict = self.dataset.get_data(obj_id, seed)

        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            trajectory=torch.from_numpy(data_dict["trajectory"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
        return cast(FlowTrajectoryTGData, data)
