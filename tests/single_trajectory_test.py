# Check len=1 trajectory should produce the same training results as original flowbot
import lightning as L
import rpad.pyg.nets.pointnet2 as pnp
import torch
from hydra import compose, initialize

from python_ml_project_template.datasets.flow_trajectory import FlowTrajectoryDataModule
from python_ml_project_template.datasets.flowbot import FlowBotDataModule
from python_ml_project_template.models.flow_predictor import FlowPredictorTrainingModule
from python_ml_project_template.models.flow_trajectory_predictor import (
    FlowTrajectoryTrainingModule,
)


# TODO: aggregate two settings to unit_test, then also needs to change the
def test_check_single_trajectory(cfg):
    with initialize(config_path="../configs", version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="unit_test")

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
    flowbot_val_loader = flowbot_datamodule.val_dataloader()
    flowbot_unseen_loader = flowbot_datamodule.unseen_dataloader()

    # TODO: Training
    mask_channel = 1 if cfg.training.mask_input_channel else 0
    flowbot_network = pnp.PN2Dense(
        in_channels=mask_channel, out_channels=3, p=pnp.PN2DenseParams()
    )

    flowbot_model = FlowPredictorTrainingModule(
        flowbot_network, training_cfg=cfg.training
    )
    flowbot_trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision="16-mixed",
        max_epochs=2,
    )

    flowbot_trainer.fit(
        flowbot_model,
        flowbot_train_loader,
        [flowbot_train_loader, flowbot_val_loader, flowbot_unseen_loader],
    )

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
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision="16-mixed",
        max_epochs=2,
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
