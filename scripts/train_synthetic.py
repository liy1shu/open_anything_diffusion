# Simple synthetic dataset training
import json

import hydra
import lightning as L
import omegaconf
import rpad.pyg.nets.pointnet2 as pnp
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from open_anything_diffusion.datasets.synthetic_dataset import SyntheticDataModule
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffusionModule,
)
from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowTrajectoryTrainingModule,
)
from open_anything_diffusion.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    match_fn,
)

training_module_class = {
    "synthetic_artflownet": FlowTrajectoryTrainingModule,
    "synthetic_diffuser": FlowTrajectoryDiffusionModule,
}


@hydra.main(config_path="../configs", config_name="train_synthetic", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    # torch.set_float32_matmul_precision("medium")
    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    L.seed_everything(cfg.seed)

    ######################################################################
    # Create the datamodule.
    # The datamodule is responsible for all the data loading, including
    # downloading the data, and splitting it into train/val/test.
    #
    # This could be swapped out for a different datamodule in-place,
    # or with an if statement, or by using hydra.instantiate.
    ######################################################################

    trajectory_len = 1 if cfg.dataset.name == "flowbot" else cfg.training.trajectory_len
    # Create synthetic dataset
    datamodule = SyntheticDataModule(batch_size=cfg.training.batch_size, seed=cfg.seed)
    train_loader = datamodule.train_dataloader()
    train_val_loader = datamodule.train_val_dataloader()
    val_loader = datamodule.val_dataloader()
    unseen_loader = datamodule.unseen_dataloader()

    ######################################################################
    # Create the network(s) which will be trained by the Training Module.
    # The network should (ideally) be lightning-independent. This allows
    # us to use the network in other projects, or in other training
    # configurations.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `open_anything_diffusion.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).

    if cfg.model.name == "diffuser":
        in_channels = 3 * cfg.training.trajectory_len + cfg.model.time_embed_dim
    else:
        in_channels = 1 if cfg.training.mask_input_channel else 0

    network = pnp.PN2Dense(
        in_channels=in_channels,
        out_channels=3 * trajectory_len,
        p=pnp.PN2DenseParams(),
    )

    ######################################################################
    # Create the training module.
    # The training module is responsible for all the different parts of
    # training, including the network, the optimizer, the loss function,
    # and the logging.
    ######################################################################

    model = training_module_class[cfg.training.name](
        network, training_cfg=cfg.training, model_cfg=cfg.model
    )

    ######################################################################
    # Set up logging in WandB.
    # This is a bit complicated, because we want to log the codebase,
    # the model, and the checkpoints.
    ######################################################################

    # If no group is provided, then we should create a new one (so we can allocate)
    # evaluations to this group later.
    if cfg.wandb.group is None:
        id = wandb.util.generate_id()
        group = "experiment-" + id
    else:
        group = cfg.wandb.group

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    #
    # There are a few callbacks (which we could customize):
    # - LogPredictionSamplesCallback: Logs some examples from the dataset,
    #       and the model's predictions.
    # - ModelCheckpoint #1: Saves the latest model.
    # - ModelCheckpoint #2: Saves the best model (according to validation
    #       loss), and logs it to wandb.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        max_epochs=cfg.training.epochs,
        logger=logger,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        callbacks=[
            # Callback which logs whatever visuals (i.e. dataset examples, preds, etc.) we want.
            LogPredictionSamplesCallback(
                logger=logger,
                eval_per_n_epoch=cfg.training.check_val_every_n_epoch,
            ),
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
            ),
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
                monitor="val/flow_loss" if cfg.model.name == "diffuser" else "val/loss",
                mode="min",
                save_weights_only=True,
            ),
        ],
    )

    ######################################################################
    # Log the code to wandb.
    # This is somewhat custom, you'll have to edit this to include whatever
    # additional files you want, but basically it just logs all the files
    # in the project root inside dirs, and with extensions.
    ######################################################################

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Train the model.
    ######################################################################

    # trainer.fit(model, train_loader, [val_loader, train_val_loader, unseen_loader], ckpt_path='/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-09-11/19-01-57/checkpoints/last.ckpt')
    trainer.fit(model, train_loader, [val_loader, train_val_loader, unseen_loader])


if __name__ == "__main__":
    main()
