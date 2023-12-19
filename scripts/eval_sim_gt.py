# The evaluation script that runs a rollout for each object in the eval-ed dataset and calculates:
# - success : 90% open
# - distance to open
import json
import os

import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from open_anything_diffusion.simulations.simulation import trial_gt_trajectory
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


def load_obj_and_link():
    with open("./scripts/umpnet_object_list.json", "r") as f:
        object_link_json = json.load(f)
    return object_link_json


id_to_cat = load_obj_id_to_category()
object_to_link = load_obj_and_link()


@hydra.main(config_path="../configs", config_name="eval_sim_gt", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

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
    id = wandb.util.generate_id()
    group = "experiment-" + id
    # if cfg.wandb.group is None:
    #     id = wandb.util.generate_id()
    #     group = "experiment-" + id
    # else:
    #     group = cfg.wandb.group

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
        group=group,
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

    # Simulation and results.
    print("Simulating")

    if cfg.website:
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
    # for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
    for obj_id, available_links in tqdm.tqdm(list(object_to_link.items())):
        obj_cat = id_to_cat[obj_id]
        if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
            continue
        print(f"OBJ {obj_id} of {obj_cat}")
        # trial_figs, trial_results = trial_flow(
        #     obj_id=obj_id,
        #     n_steps=30,
        #     all_joint=True,
        #     available_joints=available_links,
        #     gui=cfg.gui,
        #     website=cfg.website,
        # )
        trial_figs, trial_results = trial_gt_trajectory(
            obj_id=obj_id,
            traj_len=cfg.inference.trajectory_len,
            n_steps=30,
            all_joint=True,
            available_joints=available_links,
            gui=cfg.gui,
            website=cfg.website,
        )

        if len(trial_results) == 0:  # If nothing succeeds
            continue

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

    # for obj_cat in category_counts.keys():
    #     metric_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
    #     metric_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
    #     metric_df.loc[obj_cat]["category"] = obj_cat

    # table = wandb.Table(dataframe=metric_df)
    # run.log({f"simulation_metric_table": table})


if __name__ == "__main__":
    main()
