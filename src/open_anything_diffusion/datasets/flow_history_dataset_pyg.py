from typing import Protocol, cast

import torch
import torch_geometric.data as tgd

from open_anything_diffusion.datasets.flow_history_dataset import FlowHistoryDataset
from open_anything_diffusion.datasets.flow_trajectory_dataset import (
    FlowTrajectoryDataset,
)


class FlowHistoryTGData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor  # Points in the point cloud.
    delta: torch.Tensor  # instantaneous positive 3D flow trajectories.
    point: torch.Tensor  # the trajectory waypoints
    mask: torch.Tensor  # Mask of the part of interest.

    trial_points: torch.Tensor  # A series of trial grasp points
    trial_directions: torch.Tensor  # A series of trial directions
    trial_results: torch.Tensor  # A series of trial results


class FlowHistoryPyGDataset(tgd.Dataset):
    def __init__(
        self,
        trajectory_dataset: FlowTrajectoryDataset,
        max_trial_num: int = 100,
        correct_thres: float = 0.8,
        no_history_ratio: float = 0.4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset = FlowHistoryDataset(
            trajectory_dataset, max_trial_num, correct_thres, no_history_ratio
        )
        self.seed = seed

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self.dataset._dataset._ids[index], seed=self.seed)

    @staticmethod
    def get_processed_dir(
        randomize_joints,
        randomize_camera,
        trajectory_len,
        special_req=None,
        toy_dataset_id=None,
    ):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        if special_req is None and toy_dataset_id is None:
            return f"history_{trajectory_len}_{joint_chunk}_{camera_chunk}_random"
        elif special_req is not None and toy_dataset_id is None:
            # fully_closed
            # half_half
            return (
                f"history_{trajectory_len}_{joint_chunk}_{camera_chunk}_{special_req}"
            )
        elif special_req is None and toy_dataset_id is not None:
            # fully_closed
            # half_half
            return f"history_{trajectory_len}_{joint_chunk}_{camera_chunk}_toy{toy_dataset_id}_random"
        else:
            return f"history_{trajectory_len}_{joint_chunk}_{camera_chunk}_{special_req}_toy{toy_dataset_id}"

    def get_data(self, obj_id: str, seed) -> FlowHistoryTGData:
        data_dict = self.dataset.get_data(obj_id, seed)
        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            delta=torch.from_numpy(data_dict["delta"]).float(),
            point=torch.from_numpy(data_dict["point"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
            trial_points=torch.from_numpy(data_dict["trial_points"]).float(),
            trial_directions=torch.from_numpy(data_dict["trial_directions"]).float(),
            trial_results=torch.from_numpy(data_dict["trial_results"]).float(),
        )
        return cast(FlowHistoryTGData, data)
