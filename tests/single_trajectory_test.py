# Check len=1 trajectory should produce the same training results as original flowbot
import copy
from typing import Any

import lightning as L
import rpad.pyg.nets.pointnet2 as pnp
import torch
from lightning.pytorch.utilities.seed import isolate_rng
from omegaconf import OmegaConf
from torch import Tensor

from python_ml_project_template.datasets.flow_trajectory import FlowTrajectoryDataModule
from python_ml_project_template.datasets.flowbot import FlowBotDataModule
from python_ml_project_template.models.flow_predictor import FlowPredictorTrainingModule
from python_ml_project_template.models.flow_trajectory_predictor import (
    FlowTrajectoryTrainingModule,
)


def check_model_same(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert p1.shape == p2.shape
        assert torch.equal(p1, p2), f"weight diff: {torch.abs(p1 - p2)}"


def check_data_same(bs1, bs2):
    for b1, b2 in zip(bs1, bs2):
        for v1, v2 in zip(b1.values(), b2.values()):
            if torch.is_tensor(v1):
                assert torch.equal(v1, v2), f"tensor diff: {torch.abs(v1 - v2)}"
            elif isinstance(v1, list):
                for item1, item2 in zip(v1, v2):
                    assert item1 == item2
            else:
                assert False, f"{type(v1)} is not included."


def check_loss_same(ls1, ls2):
    for i, (l1, l2) in enumerate(zip(ls1, ls2)):
        assert torch.equal(l1, l2), f"loss {i} is different"


def check_preds_same(ps1, ps2):
    for i, (p1, p2) in enumerate(zip(ps1, ps2)):
        # breakpoint()
        assert torch.equal(p1, p2), f"pred {i} is different"


def check_network_same(n1, n2):
    for v1, v2 in zip(n1.state_dict().values(), n2.state_dict().values()):
        assert torch.equal(v1, v2)


def check_sds_same(sds1, sds2):
    for sd1, sd2 in zip(sds1, sds2):
        for (k1, v1), (k2, v2) in zip(sd1.items(), sd2.items()):
            assert k1 == k2
            assert torch.equal(v1, v2), f"{k1} is not the same!"


class DatasetCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.batches = []
        self.losses = []
        self.preds = []
        self.pre_backwards_sds = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.losses.append(outputs["loss"])
        self.preds.append(outputs["preds"])

    def on_before_backward(
        self, trainer: L.Trainer, pl_module: L.LightningModule, loss: Tensor
    ) -> None:
        self.pre_backwards_sds.append(copy.deepcopy(pl_module.state_dict()))

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.batches.append(batch)


def run_training_flowbot(steps):
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "flowbot",
                "data_dir": "/home/yishu/datasets/partnet-mobility",
                "dataset": "umpnet",
                "model": "flowbot",
                "batch_size": 1,
                "lr": 1e-3,
                "mask_input_channel": True,
                "randomize_camera": True,
                "seed": 42,
            },
            "training": {
                "lr": 1e-3,
                "batch_size": 1,
                "epochs": 100,
                "n_proc": 0,
                "check_val_every_n_epoch": 1,
                "trajectory_len": 1,
                "mode": "delta",
                "mask_input_channel": True,
            },
            "resources": {
                "gpus": [1],
                "num_workers": 0,
            },
            "seed": 42,
        }
    )

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Run 2 epoch of original flowbot model
    L.seed_everything(cfg.seed)
    flowbot_datamodule = FlowBotDataModule(
        root=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        n_proc=cfg.training.n_proc,  # Add n_proc
    )
    flowbot_train_loader = flowbot_datamodule.train_dataloader()
    # flowbot_val_loader = flowbot_datamodule.val_dataloader()
    # flowbot_unseen_loader = flowbot_datamodule.unseen_dataloader()

    # TODO: Training
    L.seed_everything(cfg.seed)
    mask_channel = 1 if cfg.training.mask_input_channel else 0
    flowbot_network = pnp.PN2Dense(
        in_channels=mask_channel, out_channels=3, p=pnp.PN2DenseParams()
    )
    initial_network = copy.deepcopy(flowbot_network)

    callback = DatasetCallback()

    L.seed_everything(cfg.seed)
    flowbot_model = FlowPredictorTrainingModule(
        flowbot_network, training_cfg=cfg.training
    )
    flowbot_trainer = L.Trainer(
        accelerator="cpu",
        # devices=cfg.resources.gpus,
        # precision="16-mixed",
        max_steps=steps,
        callbacks=[callback],
        deterministic=True,
    )

    L.seed_everything(cfg.seed)
    flowbot_trainer.fit(
        flowbot_model,
        flowbot_train_loader,
        # [flowbot_train_loader, flowbot_val_loader, flowbot_unseen_loader],
    )

    return (
        flowbot_model,
        callback.batches,
        callback.losses,
        initial_network,
        flowbot_network,
        callback.preds,
        callback.pre_backwards_sds,
    )


# TODO: aggregate two settings to unit_test, then also needs to change the
def test_check_single_trajectory():
    # with initialize(config_path="../configs", version_base="1.3"):
    #     # config is relative to a module
    #     cfg = compose(config_name="unit_test")

    # cfg = OmegaConf.create({
    #     "dataset": {
    #         "name": "flowbot",
    #         "data_dir": "/home/yishu/datasets/partnet-mobility",
    #         "dataset": "umpnet",
    #         "model": "flowbot",
    #         "batch_size": 64,
    #         "lr": 1e-3,
    #         "mask_input_channel": True,
    #         "randomize_camera": True,
    #         "seed": 42,
    #     },
    #     "training": {
    #         "lr": 1e-3,
    #         "batch_size": 64,
    #         "epochs": 100,
    #         "n_proc": 0,
    #         "check_val_every_n_epoch": 1,
    #         "trajectory_len": 1,
    #         "mode": "delta",
    #         "mask_input_channel": True,
    #     },
    #     "resources": {
    #         "gpus": [1],
    #         "num_workers": 4,
    #     },
    #     "seed": 42,

    # })

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_float32_matmul_precision("medium")

    # # Run 2 epoch of original flowbot model
    # L.seed_everything(cfg.seed)
    # flowbot_datamodule = FlowBotDataModule(
    #     root=cfg.dataset.data_dir,
    #     batch_size=cfg.training.batch_size,
    #     num_workers=cfg.resources.num_workers,
    #     n_proc=cfg.training.n_proc,  # Add n_proc
    # )
    # flowbot_train_loader = flowbot_datamodule.train_dataloader()
    # flowbot_val_loader = flowbot_datamodule.val_dataloader()
    # flowbot_unseen_loader = flowbot_datamodule.unseen_dataloader()

    # # TODO: Training
    # mask_channel = 1 if cfg.training.mask_input_channel else 0
    # flowbot_network = pnp.PN2Dense(
    #     in_channels=mask_channel, out_channels=3, p=pnp.PN2DenseParams()
    # )

    # flowbot_model = FlowPredictorTrainingModule(
    #     flowbot_network, training_cfg=cfg.training
    # )
    # flowbot_trainer = L.Trainer(
    #     accelerator="gpu",
    #     devices=cfg.resources.gpus,
    #     precision="16-mixed",
    #     max_steps=5,
    # )

    # flowbot_trainer.fit(
    #     flowbot_model,
    #     flowbot_train_loader,
    #     [flowbot_train_loader, flowbot_val_loader, flowbot_unseen_loader],
    # )

    with isolate_rng(True):
        m1, b1, l1, n0i, n0f, p1, pbsd1 = run_training_flowbot(1)
    with isolate_rng(True):
        m2, b2, l2, n1i, n1f, p2, pbsd2 = run_training_flowbot(1)
    check_data_same(b1, b2)

    # Check deterministic prediction...
    # preds = []
    # with isolate_rng(True):
    #     pred_0 = n0i(b1[0])
    # with isolate_rng(True):
    #     pred_1 = n1i(b1[0])

    check_network_same(n0i, n1i)

    check_sds_same(pbsd1, pbsd2)

    check_preds_same(p1, p2)
    check_loss_same(l1, l2)
    check_network_same(n0f, n1f)
    check_model_same(m1, m2)

    breakpoint()

    # Run 2 epoch of 1-trajectory model
    L.seed_everything(cfg.seed)
    # TODO: Dataset
    traj_datamodule = FlowTrajectoryDataModule(
        root=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        trajectory_len=cfg.training.trajectory_len,
        n_proc=cfg.training.n_proc,  # Add n_proc
        seed=cfg.seed,
    )
    traj_train_loader = traj_datamodule.train_dataloader()
    traj_val_loader = traj_datamodule.val_dataloader()
    traj_unseen_loader = traj_datamodule.unseen_dataloader()

    mask_channel = 1 if cfg.training.mask_input_channel else 0
    traj_network = pnp.PN2Dense(
        in_channels=mask_channel,
        out_channels=3 * 1,
        p=pnp.PN2DenseParams(),
    )

    traj_trainer = L.Trainer(
        accelerator="cpu",
        # devices=cfg.resources.gpus,
        # precision="16-mixed",
        max_steps=10,
        callbacks=[],
    )
    traj_model = FlowTrajectoryTrainingModule(traj_network, training_cfg=cfg.training)
    traj_trainer.fit(
        traj_model,
        traj_train_loader,
        [traj_train_loader, traj_val_loader, traj_unseen_loader],
    )

    # Compare the weights: are they the same (diff < eps)
    for p1, p2 in zip(flowbot_model.parameters(), traj_model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
