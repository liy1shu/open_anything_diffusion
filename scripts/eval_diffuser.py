# Diffuser evaluation scripts
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import pandas as pd
import rpad.partnet_mobility_utils.dataset as rpd
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
import wandb

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule
from open_anything_diffusion.metrics.trajectory import (
    flow_metrics,
    normalize_trajectory,
)
from open_anything_diffusion.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffuserInferenceModule_DGDiT,
)
from open_anything_diffusion.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserInferenceModule_DiT,
)
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserInferenceModule_PN2,
)
from open_anything_diffusion.utils.script_utils import PROJECT_ROOT, match_fn

data_module_class = {
    "trajectory": FlowTrajectoryDataModule,
}

inference_module_class = {
    "trajectory_diffuser_pn++": FlowTrajectoryDiffuserInferenceModule_PN2,
    "trajectory_diffuser_dgdit": FlowTrajectoryDiffuserInferenceModule_DGDiT,
    "trajectory_diffuser_dit": FlowTrajectoryDiffuserInferenceModule_DiT,
}


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    L.seed_everything(42)

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################
    trajectory_len = cfg.inference.trajectory_len
    # Create FlowBot dataset
    datamodule = data_module_class[cfg.dataset.name](
        root=cfg.dataset.data_dir,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.resources.num_workers,
        n_proc=cfg.resources.n_proc_per_worker,
        seed=cfg.seed,
        trajectory_len=trajectory_len,  # Only used when inference trajectory model
        toy_dataset={
            "id": "door-full-new",
            "train-train": [
                "8877",
                "8893",
                "8897",
                "8903",
                "8919",
                "8930",
                "8961",
                "8997",
                "9016",
                "9032",
                "9035",
                "9041",
                "9065",
                "9070",
                "9107",
                "9117",
                "9127",
                "9128",
                "9148",
                "9164",
                "9168",
                "9277",
                "9280",
                "9281",
                "9288",
                "9386",
                "9388",
                "9410",
            ],
            "train-test": [
                "8877",
                "8893",
                "8897",
                "8903",
                "8919",
                "8930",
                "8961",
                "8997",
                "9016",
                "9032",
                "9035",
                "9041",
                "9065",
                "9070",
                "9107",
                "9117",
                "9127",
                "9128",
                "9148",
                "9164",
                "9168",
                "9277",
                "9280",
                "9281",
                "9288",
                "9386",
                "9388",
                "9410",
            ],
            "test": [
                "8877",
                "8893",
                "8897",
                "8903",
                "8919",
                "8930",
                "8961",
                "8997",
                "9016",
                "9032",
                "9035",
                "9041",
                "9065",
                "9070",
                "9107",
                "9117",
                "9127",
                "9128",
                "9148",
                "9164",
                "9168",
                "9277",
                "9280",
                "9281",
                "9288",
                "9386",
                "9388",
                "9410",
            ],
        },
    )

    ######################################################################
    # Set up logging in WandB.
    # This is a different job type (eval), but we want it all grouped
    # together. Notice that we use our own logging here (not lightning).
    ######################################################################

    # Create a run.
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=cfg.wandb.group,
    )

    # Log the code.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    in_channels = 3 * cfg.inference.trajectory_len + cfg.model.time_embed_dim
    network = pnp.PN2Dense(
        in_channels=in_channels,
        out_channels=3 * trajectory_len,
        p=pnp.PN2DenseParams(),
    )

    # Get the checkpoint file. If it's a wandb reference, download.
    # Otherwise look to disk.
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = cfg.wandb.artifact_dir
        artifact = run.use_artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # ckpt_file = '/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-09-11/19-08-26/checkpoints/epoch=5004-step=3933930.ckpt'

    # # Load the network weights.
    # ckpt = torch.load(ckpt_file)
    # network.load_state_dict(
    #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    # )

    ######################################################################
    # Create an inference module, which is basically just a bare-bones
    # class which runs the model. In this example, we only implement
    # the "predict_step" function, which may not be the blessed
    # way to do it vis a vis lightning, but whatever.
    #
    # If this is a downstream application or something, you might
    # want to implement a different interface (like with a "predict"
    # function), so you can pass in un-batched observations from an
    # environment, for instance.
    ######################################################################

    model = inference_module_class[cfg.dataset.name](
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    )
    model.load_from_ckpt(ckpt_file)
    model.eval()

    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision="32-true",
        logger=False,
    )

    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################

    dataloaders = [
        (datamodule.train_val_dataloader(), "train"),
        (datamodule.val_dataloader(), "val"),
        (datamodule.unseen_dataloader(), "test"),
    ]

    all_objs = (
        rpd.UMPNET_TRAIN_TRAIN_OBJS + rpd.UMPNET_TRAIN_TEST_OBJS + rpd.UMPNET_TEST_OBJS
    )
    id_to_obj_class = {obj_id: obj_class for obj_id, obj_class in all_objs}

    for loader, name in dataloaders:
        metrics = []
        outputs = trainer.predict(
            model,
            dataloaders=[loader],
        )

        for batch, preds in zip(tqdm.tqdm(loader), outputs):
            st = 0
            for data in batch.to_data_list():
                f_pred = preds[st : st + data.num_nodes]
                f_pred = f_pred.reshape(f_pred.shape[0], -1, 3)

                # Ignore nan predictions for now...
                if torch.isnan(f_pred).sum() != 0:
                    continue

                f_ix = data.mask.bool()
                if cfg.dataset.name == "trajectory":
                    f_target = data.delta
                else:
                    f_target = data.flow
                    f_target = f_target.reshape(f_target.shape[0], -1, 3)

                f_pred = normalize_trajectory(f_pred)
                f_target = normalize_trajectory(f_target)
                rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

                metrics.append(
                    {
                        "id": data.id,
                        "obj_class": id_to_obj_class[data.id],
                        "metrics": {
                            "rmse": rmse.cpu().item(),
                            "cos_dist": cos_dist.cpu().item(),
                            "mag_error": mag_error.cpu().item(),
                        },
                    }
                )

                st += data.num_nodes

        rows = [
            (
                m["id"],
                m["obj_class"],
                m["metrics"]["rmse"],
                m["metrics"]["cos_dist"],
                m["metrics"]["mag_error"],
            )
            for m in metrics
        ]
        raw_df = pd.DataFrame(
            rows, columns=["id", "category", "rmse", "cos_dist", "mag_error"]
        )
        df = raw_df.groupby("category").mean(numeric_only=True)
        df.loc["unweighted_mean"] = raw_df.mean(numeric_only=True)
        df.loc["class_mean"] = df.mean()

        out_file = Path(cfg.log_dir) / f"{cfg.dataset.name}_{trajectory_len}_{name}.csv"
        print(out_file)
        # if out_file.exists():
        #     raise ValueError(f"{out_file} already exists...")
        df.to_csv(out_file, float_format="%.3f")

        # Log the metrics + table to wandb.
        table = wandb.Table(dataframe=df.reset_index())
        run.log({f"{name}_metric_table": table})


if __name__ == "__main__":
    main()
