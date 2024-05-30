import json

import hydra
import lightning as L
import omegaconf

# Modules
import rpad.pyg.nets.pointnet2 as pnp
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule
from open_anything_diffusion.datasets.flowbot import FlowBotDataModule
from open_anything_diffusion.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffusionModule_DGDiT,
)
from open_anything_diffusion.models.flow_diffuser_dit import (
    FlowTrajectoryDiffusionModule_DiT,
)
from open_anything_diffusion.models.flow_diffuser_hisdit import (
    FlowTrajectoryDiffusionModule_HisDiT,
)
from open_anything_diffusion.models.flow_diffuser_hispndit import (
    FlowTrajectoryDiffusionModule_HisPNDiT,
)
from open_anything_diffusion.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffusionModule_PNDiT,
)

# Regression Models
from open_anything_diffusion.models.flow_predictor import FlowPredictorTrainingModule

# Diffusion Models
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffusionModule_PN2,
)
from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowTrajectoryTrainingModule,
)
from open_anything_diffusion.models.modules.dit_models import (
    DGDiT,
    DiT,
    PN2DiT,
    PN2HisDiT,
)
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder
from open_anything_diffusion.models.modules.history_translator import HistoryTranslator
from open_anything_diffusion.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    match_fn,
)

data_module_class = {
    "flowbot": FlowBotDataModule,
    "trajectory": FlowTrajectoryDataModule,
}
training_module_class = {
    "flowbot_pn++": FlowPredictorTrainingModule,
    "trajectory_pn++": FlowTrajectoryTrainingModule,
    "trajectory_diffuser_pn++": FlowTrajectoryDiffusionModule_PN2,
    "trajectory_diffuser_pndit": FlowTrajectoryDiffusionModule_PNDiT,
    "trajectory_diffuser_dgdit": FlowTrajectoryDiffusionModule_DGDiT,
    "trajectory_diffuser_dit": FlowTrajectoryDiffusionModule_DiT,
    # With history
    "trajectory_diffuser_hisdit": FlowTrajectoryDiffusionModule_HisDiT,
    "trajectory_diffuser_hispndit": FlowTrajectoryDiffusionModule_HisPNDiT,
}
history_network_class = {
    "encoder": HistoryEncoder,
    "translator": HistoryTranslator,
}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
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
    # 1) toy_dataset = None
    # 2) toy_dataset = {
    #     "id": "door-1",
    #     "train-train": ["8994", "9035"],
    #     "train-test": ["8994", "9035"],
    #     "test": ["8867"],
    #     # "train-train": ["8867"],
    #     # "train-test": ["8867"],
    #     # "test": ["8867"],
    # }
    # 3) toy_dataset = {
    #     "id": "door-full",
    #     "train-train": ["8867", "8877", "8893", "8897", "8903", "8919", "8930", "8936", "8961", "8983", "8994", "8997", "9003", "9016", "9032", "9035", "9041", "9065", "9070", "9107", "9117", "9127", "9128", "9148", "9164", "9168", "9263", "9277", "9280", "9281", "9288", "9386", "9388", "9393", "9410"],
    #     "train-test": ["8994"],
    #     "test": ["9035"],
    # }

    # Full dataset
    toy_dataset = None
    # # Door dataset
    # toy_dataset = {
    #     "id": "door-full-new",
    #     "train-train": [
    #         "8877",
    #         "8893",
    #         "8897",
    #         "8903",
    #         "8919",
    #         "8930",
    #         "8961",
    #         "8997",
    #         "9016",
    #         "9032",
    #         "9035",
    #         "9041",
    #         "9065",
    #         "9070",
    #         "9107",
    #         "9117",
    #         "9127",
    #         "9128",
    #         "9148",
    #         "9164",
    #         "9168",
    #         "9277",
    #         "9280",
    #         "9281",
    #         "9288",
    #         "9386",
    #         "9388",
    #         "9410",
    #     ],
    #     "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
    #     "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
    # }
    # special_req = "half-half"  # "fully-closed"
    special_req = "50-150" #"half-half" #None #"half-half"

    # Create flow dataset
    datamodule = data_module_class[cfg.dataset.name](
        root=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        n_proc=cfg.resources.n_proc_per_worker,
        seed=cfg.seed,
        history="his" in cfg.model.name,
        trajectory_len=trajectory_len,  # Only used when training trajectory model
        history_len=cfg.training.history_len,
        special_req=special_req,  # special_req="fully-closed"
        # # TODO: only for toy training!!!!!
        toy_dataset=toy_dataset,
    )
    train_loader = datamodule.train_dataloader()
    if "diffuser" in cfg.model.name:
        cfg.training.train_sample_number = len(train_loader)
    eval_sample_bsz = 1 if cfg.training.wta else cfg.training.batch_size
    train_val_loader = datamodule.train_val_dataloader(bsz=eval_sample_bsz)

    if special_req == "half-half" and toy_dataset is not None:  # half-half doors
        # For half-half training:
        # - Unseen loader: randomly opened doors
        # - Validation loader: fully closed doors
        randomly_opened_datamodule = data_module_class[cfg.dataset.name](
            root=cfg.dataset.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.resources.num_workers,
            n_proc=cfg.resources.n_proc_per_worker,
            seed=cfg.seed,
            history="his" in cfg.model.name,
            trajectory_len=trajectory_len,  # Only used when training trajectory model
            special_req=None,  # special_req="fully-closed"
            toy_dataset=toy_dataset,
        )
        fully_closed_datamodule = data_module_class[cfg.dataset.name](
            root=cfg.dataset.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.resources.num_workers,
            n_proc=cfg.resources.n_proc_per_worker,
            seed=cfg.seed,
            history="his" in cfg.model.name,
            trajectory_len=trajectory_len,  # Only used when training trajectory model
            special_req="fully-closed",  # special_req="fully-closed"
            toy_dataset=toy_dataset,
        )
        val_loader = fully_closed_datamodule.val_dataloader(bsz=eval_sample_bsz)
        unseen_loader = randomly_opened_datamodule.unseen_dataloader(
            bsz=eval_sample_bsz
        )
    else:  # half-half full dataset
        val_loader = datamodule.val_dataloader(bsz=eval_sample_bsz)
        unseen_loader = datamodule.unseen_dataloader(bsz=eval_sample_bsz)

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

    if "diffuser" in cfg.model.name:
        if "pn++" in cfg.model.name:
            in_channels = 3 * cfg.training.trajectory_len + cfg.model.time_embed_dim
        else:
            in_channels = (
                3 * cfg.training.trajectory_len
            )  # Will add 3 as input channel in diffuser
    else:
        in_channels = 1 if cfg.training.mask_input_channel else 0

    if "pn++" in cfg.model.name:
        network = pnp.PN2Dense(
            in_channels=in_channels,
            out_channels=3 * trajectory_len,
            p=pnp.PN2DenseParams(),
        ).cuda()
    elif "dgdit" in cfg.model.name:
        network = DGDiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=cfg.dataset.n_points,
        ).cuda()
    elif "hisdit" in cfg.model.name:
        network = {
            "DiT": DiT(
                in_channels=in_channels + 3 + cfg.model.history_dim,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": history_network_class[cfg.model.history_model](
                history_dim=cfg.model.history_dim,
                history_len=cfg.model.history_len,
                batch_norm=cfg.model.batch_norm,
                repeat_dim=True,
            ).cuda(),
        }
    elif "hispndit" in cfg.model.name:
        network = {
            "DiT": PN2HisDiT(
                history_embed_dim=cfg.model.history_dim,
                in_channels=in_channels,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": history_network_class[cfg.model.history_model](
                history_dim=cfg.model.history_dim,
                history_len=cfg.model.history_len,
                batch_norm=cfg.model.batch_norm,
                transformer=False,
                repeat_dim=False,
            ).cuda(),
        }
    elif "pndit" in cfg.model.name:
        network = PN2DiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=cfg.dataset.n_points,
        )
    elif "dit" in cfg.model.name:
        network = DiT(
            in_channels=in_channels + 3,
            # depth=5,
            # hidden_size=128,
            # num_heads=4,
            depth=12,
            hidden_size=384,
            num_heads=6,
            learn_sigma=True,
        ).cuda()

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
                eval_dataloader_lengths=[
                    len(val_loader),
                    len(train_val_loader),
                    len(unseen_loader),
                ],
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
                monitor="val_wta/flow_loss" if cfg.training.wta else "val/flow_loss",
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
    # breakpoint()
    # trainer.fit(model, train_loader, [val_loader, train_val_loader, unseen_loader], ckpt_path='/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-09-11/19-01-57/checkpoints/last.ckpt')
    trainer.fit(model, train_loader, [val_loader, unseen_loader]) #train_val_loader


if __name__ == "__main__":
    main()
