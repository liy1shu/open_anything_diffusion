# Generate the trial datasets:
# - Trial description: Grasp point, flow direction
# - Trial result: Cosine distance with gt (Currently use cosine, later could use more physical feedback)

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from open_anything_diffusion.datasets.flow_trajectory_dataset import (
    FlowTrajectoryDataset,
)


class FlowHistoryData(TypedDict):
    id: str
    pos: npt.NDArray[np.float32]  # (N, 3): Point cloud observation.
    delta: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth flow.
    # point: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth waypoints.
    mask: npt.NDArray[np.bool_]  #  (N,): Mask the point of interest.

    # History Trials
    trial_points: npt.NDArray[np.float32]
    trial_directions: npt.NDArray[np.float32]
    trial_results: npt.NDArray[np.float32]


class FlowHistoryDataset:
    def __init__(
        self,
        trajectory_dataset: FlowTrajectoryDataset,
        max_trial_num: int = 100,
        correct_thres: float = 0.8,
        no_history_ratio: float = 0.4,  # For inference, history_ratio is 0
    ) -> None:
        self._dataset = trajectory_dataset
        self.max_trial_num = max_trial_num
        self.n_points = self._dataset.n_points
        self.correct_thres = correct_thres
        self.no_history_ratio = no_history_ratio

    def get_data(self, obj_id: str, seed=None) -> FlowHistoryData:
        _data = self._dataset.get_data(obj_id, seed)

        # Generate the history trials here
        if np.random.choice(10) < int(
            self.no_history_ratio * 10
        ):  # 0.4 probability with no history
            return {
                "id": _data["id"],
                "pos": _data["pos"].numpy(),
                "delta": _data["delta"].numpy(),  #  N , traj_len, 3
                "point": _data["point"].numpy(),  #  N , traj_len, 3
                "mask": _data["mask"].numpy(),
                "trial_points": np.zeros((1, 3)),
                "trial_directions": np.zeros((1, 3)),
                "trial_results": np.zeros(1),
            }
        else:
            # 0) Trial counts
            trial_count = np.random.choice(self.max_trial_num - 1) + 1  # Not zero
            # 1) Grasp point
            grasp_point_idx = np.random.choice(self.n_points, size=(trial_count))
            grasp_point = _data["pos"][grasp_point_idx].numpy()
            # 2) Flow direction
            grasp_direction = np.random.randn(*grasp_point.shape)
            grasp_gt_direction = _data["delta"][grasp_point_idx].reshape(
                grasp_direction.shape[0], -1
            )
            # 3) Pseudo-Trial results
            pseudo_results = np.sum(
                grasp_direction * grasp_gt_direction.numpy(), axis=-1
            )

            # Adjust the trials:
            # 1) Only keep cosine < 0.8
            # indices = np.where(pseudo_results < self.correct_thres)
            indices = np.where(pseudo_results < 100)
            # 2) For cosine < 0, the result is cosine = 0 (won't move, the model won't distinguish completely wrong directions)
            # pseudo_results = np.where(pseudo_results < -0.2, -0.2, pseudo_results)

            return {
                "id": _data["id"],
                "pos": _data["pos"].numpy(),
                "delta": _data["delta"].numpy(),  #  N , traj_len, 3
                "point": _data["point"].numpy(),  #  N , traj_len, 3
                "mask": _data["mask"].numpy(),
                "trial_points": grasp_point[indices],
                "trial_directions": grasp_direction[indices],
                "trial_results": pseudo_results[indices],
            }

    def __getitem__(self, item: int) -> FlowHistoryData:
        obj_id = self._dataset._ids[item]
        return self.get_data(obj_id)

    def __len__(self):
        return len(self._dataset)
