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
import plotly.graph_objects as go
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from open_anything_diffusion.simulations.simulation import trial_with_prediction
from open_anything_diffusion.utils.script_utils import PROJECT_ROOT, match_fn


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"{PROJECT_ROOT}/scripts/umpnet_data_split.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for _, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = category

    else:
        with open(f"{PROJECT_ROOT}/scripts/umpnet_object_list.json", "r") as f:
            data = json.load(f)
        for split in ["train-train", "train-test"]:
            for id in toy_dataset[split]:
                id_to_cat[id] = split
    return id_to_cat


def load_obj_and_link(id_to_cat):
    # with open("./scripts/umpnet_object_list.json", "r") as f:
    with open(f"{PROJECT_ROOT}/scripts/movable_links_001.json", "r") as f:
        object_link_json = json.load(f)
    for id in id_to_cat.keys():
        if id not in object_link_json.keys():
            object_link_json[id] = []
    return object_link_json


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
toy_dataset = {  # For half-half fullset training
    "id": "door-full-fullset",
    "train-train": [
        "9281",
        "9107",
        "8997",
        "9280",
        "9070",
        "8919",
        "9168",
        "8983",
        "9016",
        "9117",
        "9041",
        "9164",
        "8936",
        "8897",
        "9386",
        "9288",
        "8903",
        "9128",
        "8930",
        "8961",
        "9003",
    ],
    "train-test": ["9065", "8867", "9410", "9388", "8893", "8877"],
    "test": ["9065", "8867", "9410", "9388", "8893", "8877"],
}
id_to_cat = load_obj_id_to_category(toy_dataset)
object_to_link = load_obj_and_link(id_to_cat)

object_ids = [  # Door
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
    "8867",
    "8983",
    "8994",
    "9003",
    "9263",
    "9393",
]


@hydra.main(config_path="../configs", config_name="eval_sim", version_base="1.3")
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

    mask_channel = 1 if cfg.inference.mask_input_channel else 0
    network = pnp.PN2Dense(
        in_channels=mask_channel,
        out_channels=3 * cfg.inference.trajectory_len,
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

    # Load the network weights.
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    network.eval()

    # Simulation and results.
    print("Simulating")

    if cfg.website:
        # Visualization html
        os.makedirs("./logs/simu_eval/video_assets/")
        doc = html.PlotlyWebsiteBuilder("Simulation Visualizations")

    obj_cats = list(set(id_to_cat.values()))
    metric_df = pd.DataFrame(
        np.zeros((len(set(id_to_cat.values())), 4)),
        index=obj_cats,
        columns=["obj_cat", "count", "success_rate", "norm_dist"],
    )
    category_counts = {}
    sim_trajectories = []
    link_names = []
    # for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
    for obj_id, available_links in tqdm.tqdm(list(object_to_link.items())):
        if obj_id not in object_ids:  # For Door dataset
            continue
        if obj_id not in id_to_cat.keys():
            continue
        if len(available_links) == 0:
            continue

        obj_cat = id_to_cat[obj_id]
        if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
            continue
        print(f"OBJ {obj_id} of {obj_cat}")
        trial_figs, trial_results, sim_trajectory = trial_with_prediction(
            obj_id=obj_id,
            network=network,
            n_step=30,
            gui=cfg.gui,
            all_joint=True,
            available_joints=available_links,
            website=cfg.website,
        )
        sim_trajectories += sim_trajectory
        link_names += [f"{obj_id}_{link}" for link in available_links]

        # breakpoint()
        if len(trial_results) == 0:  # If nothing succeeds
            continue

        # Wandb table
        if obj_cat not in category_counts.keys():
            category_counts[obj_cat] = 0
        # category_counts[obj_cat] += len(trial_results)
        for result in trial_results:
            if result.contact == False:
                continue
            category_counts[obj_cat] += 1
            metric_df.loc[obj_cat]["success_rate"] += result.success
            metric_df.loc[obj_cat]["norm_dist"] += result.metric

        if cfg.website:
            # Website visualization
            for id, (joint_name, fig) in enumerate(trial_figs.items()):
                tag = f"{obj_id}_{joint_name}"
                if fig is not None:
                    doc.add_plot(obj_cat, tag, fig)
                doc.add_video(
                    obj_cat,
                    f"{tag}{'_NO CONTACT' if not trial_results[id].contact else ''}",
                    f"http://128.2.178.238:{cfg.website_port}/video_assets/{tag}.mp4",
                )
            # print(trial_results)
            doc.write_site("./logs/simu_eval")

        if category_counts[obj_cat] == 0:
            continue
        wandb_df = metric_df.copy(deep=True)
        for obj_cat in category_counts.keys():
            wandb_df.loc[obj_cat]["obj_cat"] = 0 if "train" in obj_cat else 1
            wandb_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["count"] = category_counts[obj_cat]

        # table = wandb.Table(dataframe=wandb_df.reset_index())
        table = wandb.Table(dataframe=wandb_df)
        run.log({f"simulation_metric_table": table})

    print(wandb_df)
    # for obj_cat in category_counts.keys():
    #     metric_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
    #     metric_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
    #     metric_df.loc[obj_cat]["category"] = obj_cat

    # table = wandb.Table(dataframe=metric_df)
    # run.log({f"simulation_metric_table": table})
    traces = []
    xs = list(range(31))
    for id, sim_trajectory in enumerate(sim_trajectories):
        traces.append(
            go.Scatter(x=xs, y=sim_trajectory, mode="lines", name=link_names[id])
        )

    layout = go.Layout(title="Simulation Trajectory Figure")
    fig = go.Figure(data=traces, layout=layout)
    wandb.log({"sim_traj_figure": wandb.Plotly(fig)})


if __name__ == "__main__":
    main()
