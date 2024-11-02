from typing import Optional, Protocol, TypedDict, cast

import lightning as L
import numpy as np
import numpy.typing as npt
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl


class SyntheticData(TypedDict):
    id: int
    pos: npt.NDArray[np.float32]  # (N, 3): Point cloud observation.
    delta: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth flow.
    point: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth waypoints.
    mask: npt.NDArray[np.bool_]  #  (N,): Mask the point of interest.


class SyntheticDataset:
    def __init__(
        self,
        n_points: Optional[int] = None,
    ) -> None:
        self.n_points = n_points

        self.rectangle_pts = []
        # Rectangle
        for i in range(30):
            for j in range(40):
                self.rectangle_pts.append([i / 30, j / 40, 0.5])

        self.rectangle_pts = np.array(self.rectangle_pts)

        self.directions = np.array([[0, 0, 1], [0, 0, 1]])

    def get_data(self, id: int, seed=None) -> SyntheticData:
        pos = self.rectangle_pts
        flow = np.repeat(
            np.expand_dims(self.directions[id, :], axis=0), self.n_points, axis=0
        )
        point = pos + flow
        mask = np.ones(self.n_points)
        return {
            "id": id,
            "pos": pos,
            "delta": np.expand_dims(flow, axis=1),  #  N , traj_len, 3
            "point": np.expand_dims(point, axis=1),
            "mask": mask,
        }

    def __getitem__(self, id: int) -> SyntheticData:
        return self.get_data(id)

    def __len__(self):
        return 2


class SyntheticTGData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor  # Points in the point cloud.
    delta: torch.Tensor  # instantaneous positive 3D flow trajectories.
    point: torch.Tensor  # the trajectory waypoints
    mask: torch.Tensor  # Mask of the part of interest.


class SyntheticPyGDataset(tgd.Dataset):
    def __init__(
        self,
        n_points: Optional[int] = 1200,
        seed: int = 42,  # Randomize everything
    ) -> None:
        super().__init__()
        self.dataset = SyntheticDataset(n_points)
        self.n_points = n_points
        self.seed = seed

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(index, seed=self.seed)

    @staticmethod
    def get_processed_dir():
        return "synthetic"

    def get_data(self, obj_id: str, seed) -> SyntheticTGData:
        data_dict = self.dataset.get_data(obj_id, seed)
        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.from_numpy(data_dict["pos"]).float(),
            delta=torch.from_numpy(data_dict["delta"]).float(),
            point=torch.from_numpy(data_dict["point"]).float(),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
        return cast(SyntheticTGData, data)


# Create Synthetic datamodule
class SyntheticDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.train_dset = SyntheticPyGDataset()
        self.train_val_dset = SyntheticPyGDataset()
        self.val_dset = SyntheticPyGDataset()
        self.unseen_dset = SyntheticPyGDataset()

    def train_dataloader(self):
        L.seed_everything(self.seed)
        return tgl.DataLoader(
            self.train_dset, self.batch_size, shuffle=True, num_workers=0
        )

    def train_val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(self.train_val_dset, bsz, shuffle=False, num_workers=0)

    def val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(self.val_dset, bsz, shuffle=False, num_workers=0)

    def unseen_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(self.unseen_dset, bsz, shuffle=False, num_workers=0)
