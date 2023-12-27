# The evaluation script that runs a rollout for each object in the eval-ed dataset and calculates:
# - success : 90% open
# - distance to open

import json

# TODO:
import os

import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserSimulationModule,
)
from open_anything_diffusion.simulations.simulation import trial_with_diffuser
from open_anything_diffusion.utils.script_utils import PROJECT_ROOT, match_fn


def load_obj_id_to_category():
    # Extract existing classes.
    with open("./scripts/umpnet_data_split.json", "r") as f:
        data = json.load(f)

    id_to_cat = {}
    for _, category_dict in data.items():
        for category, split_dict in category_dict.items():
            for _, id_list in split_dict.items():
                for id in id_list:
                    id_to_cat[id] = category
    return id_to_cat


id_to_cat = load_obj_id_to_category()


@hydra.main(config_path="../configs", config_name="eval_sim", version_base="1.3")
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
    # datamodule = FlowBotDataModule(
    #     root=cfg.dataset.data_dir,
    #     batch_size=cfg.inference.batch_size,
    #     num_workers=cfg.resources.num_workers,
    #     n_proc=cfg.resources.n_proc_per_worker,  # Add n_proc
    # )

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

    trajectory_len = cfg.inference.trajectory_len
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

    # # Load the network weights.
    # ckpt = torch.load(ckpt_file)
    # network.load_state_dict(
    #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    # )
    model = FlowTrajectoryDiffuserSimulationModule(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()

    # Simulation and results.
    print("Simulating")
    # Visualization html
    os.makedirs("./logs/simu_eval/video_assets/")
    doc = html.PlotlyWebsiteBuilder("Simulation Visualizations")
    obj_cats = list(set(id_to_cat.values()))
    metric_df = pd.DataFrame(
        np.zeros((len(set(id_to_cat.values())), 3)),
        index=obj_cats,
        columns=["count", "success_rate", "norm_dist"],
    )
    category_counts = {}
    for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
        if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
            continue
        print(f"OBJ {obj_id} of {obj_cat}")
        trial_figs, trial_results = trial_with_diffuser(
            obj_id=obj_id,
            model=model,
            n_step=30,
            gui=False,
            website=False,
            all_joint=True,
        )

        # Wandb table
        if obj_cat not in category_counts.keys():
            category_counts[obj_cat] = 0
        category_counts[obj_cat] += len(trial_results)
        for result in trial_results:
            metric_df.loc[obj_cat]["success_rate"] += result.success
            metric_df.loc[obj_cat]["norm_dist"] += result.metric

        if cfg.website:
            # Website visualization
            for joint_name, fig in trial_figs.items():
                tag = f"{obj_id}_{joint_name}"
                doc.add_plot(obj_cat, tag, fig)
                doc.add_video(
                    obj_cat, tag, f"http://128.2.178.238:9000/video_assets/{tag}.mp4"
                )
            # print(trial_results)
            doc.write_site("./logs/simu_eval")

        wandb_df = metric_df.copy(deep=True)
        for obj_cat in category_counts.keys():
            wandb_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["count"] = category_counts[obj_cat]

        table = wandb.Table(dataframe=wandb_df.reset_index())
        run.log({f"simulation_metric_table": table})


if __name__ == "__main__":
    main()
