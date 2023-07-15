import os

import lightning as L
import rpad.partnet_mobility_utils.dataset as rpd
import torch_geometric.loader as tgl
from rpad.pyg.dataset import CachedByKeyDataset

from python_ml_project_template.datasets.flow_trajectory_dataset_pyg import (
    FlowTrajectoryPyGDataset,
)


# Create FlowBot datamodule
class FlowTrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        n_proc,
        randomize_camera: bool = True,
        trajectory_len: int = 5,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_dset = CachedByKeyDataset(
            dset_cls=FlowTrajectoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-train-train",
                randomize_camera=randomize_camera,
                trajectory_len=trajectory_len,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS,
            root=root,
            processed_dirname=FlowTrajectoryPyGDataset.get_processed_dir(
                True, randomize_camera, trajectory_len
            ),
            n_repeat=100,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
        )

        self.val_dset = CachedByKeyDataset(
            dset_cls=FlowTrajectoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-train-test",
                randomize_camera=randomize_camera,
                trajectory_len=trajectory_len,
            ),
            data_keys=rpd.UMPNET_TRAIN_TEST_OBJ_IDS,
            root=root,
            processed_dirname=FlowTrajectoryPyGDataset.get_processed_dir(
                True, randomize_camera, trajectory_len
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
        )

        self.unseen_dset = CachedByKeyDataset(
            dset_cls=FlowTrajectoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-test",
                randomize_camera=randomize_camera,
                trajectory_len=trajectory_len,
            ),
            data_keys=rpd.UMPNET_TEST_OBJ_IDS,
            root=root,
            processed_dirname=FlowTrajectoryPyGDataset.get_processed_dir(
                True, randomize_camera, trajectory_len
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
        )

    def train_dataloader(self):
        return tgl.DataLoader(
            self.train_dset, self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return tgl.DataLoader(
            self.val_dset, self.batch_size, shuffle=False, num_workers=0
        )

    def unseen_dataloader(self):
        return tgl.DataLoader(
            self.unseen_dset, self.batch_size, shuffle=False, num_workers=0
        )
