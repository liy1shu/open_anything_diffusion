# Diffuser evaluation scripts

import hydra
import lightning as L
import omegaconf
import pandas as pd
import rpad.pyg.nets.pointnet2 as pnp
import torch
import wandb

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule
from open_anything_diffusion.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffuserInferenceModule_DGDiT,
)
from open_anything_diffusion.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserInferenceModule_DiT,
)
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserInferenceModule_PN2,
)
from open_anything_diffusion.models.modules.dit_models import DGDiT, DiT
from open_anything_diffusion.utils.script_utils import PROJECT_ROOT, match_fn

data_module_class = {
    "trajectory": FlowTrajectoryDataModule,
}

inference_module_class = {
    "diffuser_pn++": FlowTrajectoryDiffuserInferenceModule_PN2,
    "diffuser_dgdit": FlowTrajectoryDiffuserInferenceModule_DGDiT,
    "diffuser_dit": FlowTrajectoryDiffuserInferenceModule_DiT,
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
    toy_dataset = {
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
        "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
        "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
    }
    # Create FlowBot dataset
    fully_closed_datamodule = FlowTrajectoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=1,
        num_workers=30,
        n_proc=2,
        seed=42,
        trajectory_len=1,  # Only used when training trajectory model
        special_req="fully-closed",
        # toy_dataset = {
        #     "id": "door-1",
        #     "train-train": ["8994", "9035"],
        #     "train-test": ["8994", "9035"],
        #     "test": ["8867"],
        #     # "train-train": ["8867"],
        #     # "train-test": ["8867"],
        #     # "test": ["8867"],
        # }
        toy_dataset=toy_dataset,
    )
    randomly_opened_datamodule = FlowTrajectoryDataModule(
        root="/home/yishu/datasets/partnet-mobility",
        batch_size=1,
        num_workers=30,
        n_proc=2,
        seed=42,
        trajectory_len=1,  # Only used when training trajectory model
        special_req=None,
        # toy_dataset = {
        #     "id": "door-1",
        #     "train-train": ["8994", "9035"],
        #     "train-test": ["8994", "9035"],
        #     "test": ["8867"],
        #     # "train-train": ["8867"],
        #     # "train-test": ["8867"],
        #     # "test": ["8867"],
        # }
        toy_dataset=toy_dataset,
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

    if "diffuser" in cfg.model.name:
        if "pn++" in cfg.model.name:
            in_channels = 3 * cfg.inference.trajectory_len + cfg.model.time_embed_dim
        else:
            in_channels = (
                3 * cfg.inference.trajectory_len
            )  # Will add 3 as input channel in diffuser
    else:
        in_channels = 1 if cfg.inference.mask_input_channel else 0

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
    elif "dit" in cfg.model.name:
        network = DiT(
            in_channels=in_channels + 3,
            depth=5,
            hidden_size=128,
            num_heads=4,
            learn_sigma=True,
        ).cuda()

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

    model = inference_module_class[cfg.model.name](
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    )
    model.load_from_ckpt(ckpt_file)
    model.eval()
    model.cuda()

    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    # trainer = L.Trainer(
    #     accelerator="gpu",
    #     devices=cfg.resources.gpus,
    #     precision="32-true",
    #     logger=False,
    # )

    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################

    dataloaders = [
        # (datamodule.train_val_dataloader(), "train"),
        (fully_closed_datamodule.val_dataloader(bsz=1), "val"),
        (randomly_opened_datamodule.unseen_dataloader(bsz=1), "test"),
    ]

    trial_time = 50

    all_metrics = []
    all_directions = []
    sample_cnt = 0
    for loader, name in dataloaders:
        sample_cnt += len(loader)

        metrics, directions = model.predict_wta(
            dataloader=loader, mode="delta", trial_times=trial_time
        )
        print(f"{name} metric:")
        print(metrics)

        all_metrics.append(metrics)
        all_directions += directions

    # # Scatter plot
    # ys = [d.item() for d in all_directions]
    # xs = sorted(list(range(sample_cnt)) * trial_time)
    # xs = [f"{x}" for x in xs]
    # colors = sorted(["red", "blue", "purple"] * trial_time) * sample_cnt
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.scatter(xs, ys, s=5, c=colors[: len(xs)])
    # plt.savefig(f"./{cfg.model.name}_cos_stats.jpeg")

    rows = [
        (
            id,
            m["rmse"],
            m["cosine_similarity"],
            m["mag_error"],
            m["multimodal"],
            m["pos@0.7"],
            m["neg@0.7"],
        )
        for id, m in zip(["fully closed", "randomly opened"], all_metrics)
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "category",
            "rmse",
            "cos_similarity",
            "mag_error",
            "multimodal",
            "pos@0.7",
            "neg@0.7",
        ],
    )

    # out_file = Path(cfg.log_dir) / f"{cfg.dataset.name}_{trajectory_len}_{name}.csv"
    # print(out_file)
    # # if out_file.exists():
    # #     raise ValueError(f"{out_file} already exists...")
    # df.to_csv(out_file, float_format="%.3f")

    # Log the metrics + table to wandb.
    table = wandb.Table(dataframe=df)
    run.log({f"eval_wta_metric_table": table})


if __name__ == "__main__":
    main()
