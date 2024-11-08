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
import rpad.pyg.nets.pointnet2 as pnp_orig
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from open_anything_diffusion.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffuserSimulationModule_DGDiT,
)
from open_anything_diffusion.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserSimulationModule_DiT,
)
from open_anything_diffusion.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffuserSimulationModule_PNDiT,
)
from open_anything_diffusion.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserSimulationModule_PN2,
)
from open_anything_diffusion.models.flow_trajectory_predictor import (
    FlowSimulationInferenceModule,
)
from open_anything_diffusion.models.modules.dit_models import (
    DGDiT,
    DiT,
    PN2DiT,
    PN2HisDiT,
)
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder
from open_anything_diffusion.simulations.simulation import trial_with_switch_models
from open_anything_diffusion.utils.script_utils import PROJECT_ROOT, match_fn

print(PROJECT_ROOT)


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"{PROJECT_ROOT}/scripts/umpnet_data_split_new.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for train_or_test, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = f"{category}_{train_or_test}"

    else:
        with open(f"{PROJECT_ROOT}/scripts/umpnet_object_list.json", "r") as f:
            data = json.load(f)
        for split in ["train-train", "train-test"]:
            # for split in ["train-test"]:
            for id in toy_dataset[split]:
                id_to_cat[id] = split
    return id_to_cat


def load_obj_and_link(id_to_cat):
    # with open("./scripts/umpnet_object_list.json", "r") as f:
    # with open(f"{PROJECT_ROOT}/scripts/movable_links_005.json", "r") as f:
    with open(f"{PROJECT_ROOT}/scripts/movable_links_fullset_000.json", "r") as f:
        object_link_json = json.load(f)
    for id in id_to_cat.keys():
        if id not in object_link_json.keys():
            object_link_json[id] = []
    return object_link_json


inference_module_class = {
    "pn++": FlowSimulationInferenceModule,
    "diffuser_pn++": FlowTrajectoryDiffuserSimulationModule_PN2,
    "diffuser_dgdit": FlowTrajectoryDiffuserSimulationModule_DGDiT,
    "diffuser_dit": FlowTrajectoryDiffuserSimulationModule_DiT,
    "diffuser_pndit": FlowTrajectoryDiffuserSimulationModule_PNDiT,
}

ckpts_dict = {
    # "pn++": "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-03-30/08-16-05/checkpoints/epoch=88-step=98345-val_loss=0.00-weights-only.ckpt",
    "pn++": "/home/yishu/open_anything_diffusion/logs/train_trajectory_pn++/2024-05-26/02-37-08/checkpoints/epoch=98-step=109395-val_loss=0.00-weights-only.ckpt",
    # "diffuser_dit": "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt",
    "diffuser_dit": "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-05-02/12-35-27/checkpoints/epoch=109-step=243100-val_loss=0.00-weights-only.ckpt",  # Large
    "diffuser_pndit": "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_pndit/2024-04-23/05-01-44/checkpoints/epoch=469-step=1038700-val_loss=0.00-weights-only.ckpt",
    "diffuser_hispndit": "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hispndit/2024-05-19/10-49-49/checkpoints/epoch=339-step=281860-val_loss=0.00-weights-only.ckpt",
    "diffuser_hisdit": "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hisdit/2024-05-10/12-09-08/checkpoints/epoch=439-step=243320-val_loss=0.00-weights-only.ckpt",
}


def create_model(inference_cfg, model_cfg, dataset_cfg, run, model_name):
    trajectory_len = inference_cfg.trajectory_len
    if "diffuser" in model_name:
        if "pn++" in model_name:
            in_channels = 3 * inference_cfg.trajectory_len + model_cfg.time_embed_dim
        else:
            in_channels = (
                3 * inference_cfg.trajectory_len
            )  # Will add 3 as input channel in diffuser
    else:
        in_channels = 1 if inference_cfg.mask_input_channel else 0

    if "pn++" in model_name:
        network = pnp_orig.PN2Dense(
            in_channels=in_channels,
            out_channels=3 * trajectory_len,
            p=pnp_orig.PN2DenseParams(),
        ).cuda()
    elif "dgdit" in model_name:
        network = DGDiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=dataset_cfg.n_points,
        ).cuda()

    if "hispndit" in model_cfg.name:
        network = {
            "DiT": PN2HisDiT(
                history_embed_dim=model_cfg.history_dim,
                in_channels=3,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": HistoryEncoder(
                history_dim=model_cfg.history_dim,
                history_len=model_cfg.history_len,
                batch_norm=model_cfg.batch_norm,
                transformer=False,
                repeat_dim=False,
            ).cuda(),
        }
        # model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
        #     network, inference_cfg=inference_cfg, model_cfg=model_cfg
        # ).cuda()
    elif "hisdit" in model_cfg.name:
        network = {
            "DiT": DiT(
                in_channels=3 + 3 + 128,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": HistoryEncoder(
                history_dim=128, history_len=1, batch_norm=False
            ).cuda(),
        }
        # model = FlowTrajectoryDiffuserSimulationModule_HisDiT(
        #     network, inference_cfg=inference_cfg, model_cfg=model_cfg
        # ).cuda()
    elif "pndit" in model_name:
        network = PN2DiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=dataset_cfg.n_points,
        ).cuda()
    elif "dit" in model_name:
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

    # ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt"
    if model_name in ckpts_dict.keys():
        ckpt_file = ckpts_dict[model_name]
    else:
        assert (
            True
        ), "No ckpt provided, currently doesn't support downloading from wandb"
        # # Get the checkpoint file. If it's a wandb reference, download.
        # # Otherwise look to disk.
        # checkpoint_reference = cfg.checkpoint.reference
        # if checkpoint_reference.startswith(cfg.wandb.entity):
        #     # download checkpoint locally (if not already cached)
        #     artifact_dir = cfg.wandb.artifact_dir
        #     artifact = run.use_artifact(checkpoint_reference, type="model")
        #     ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # else:
        #     ckpt_file = checkpoint_reference

    # # Load the network weights.
    # ckpt = torch.load(ckpt_file)
    # network.load_state_dict(
    #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    # )
    model = inference_module_class[model_cfg.name](
        network, inference_cfg=inference_cfg, model_cfg=model_cfg
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()
    return model


@hydra.main(config_path="../configs", config_name="eval_sim_switch", version_base="1.3")
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
    # toy_dataset = None   (1)
    # toy_dataset = {   # For half-half door training   (2)
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
    # toy_dataset = {  # For half-half fullset training
    #     "id": "door-full-fullset",
    #     "train-train": [
    #         "9281",
    #         "9107",
    #         "8997",
    #         "9280",
    #         "9070",
    #         "8919",
    #         "9168",
    #         "8983",
    #         "9016",
    #         "9117",
    #         "9041",
    #         "9164",
    #         "8936",
    #         "8897",
    #         "9386",
    #         "9288",
    #         "8903",
    #         "9128",
    #         "8930",
    #         "8961",
    #         "9003",
    #     ],
    #     "train-test": ["9065", "8867", "9410", "9388", "8893", "8877"],
    #     "test": ["9065", "8867", "9410", "9388", "8893", "8877"],
    # }
    toy_dataset = None  # Full dataset!
    id_to_cat = load_obj_id_to_category(toy_dataset)
    object_to_link = load_obj_and_link(id_to_cat)
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

    # Create the first model
    if cfg.model.name is not None:
        model = create_model(cfg.inference, cfg.model, cfg.dataset, run, cfg.model.name)
        history_for_model = "his" in cfg.model.name

    # Create the second model
    if cfg.switch_model.name is not None:
        switch_model = create_model(
            cfg.switch_inference,
            cfg.switch_model,
            cfg.dataset,
            run,
            cfg.switch_model.name,
        )
        history_for_switch_model = "his" in cfg.switch_model.name
    else:
        switch_model = model
        history_for_switch_model = "his" in cfg.model.name

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

    for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
        if "Door_test" not in obj_cat:
            continue
        if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
            continue
        available_links = object_to_link[obj_id]
        if len(available_links) == 0:
            continue
        print(f"OBJ {obj_id} of {obj_cat}")
        trial_figs, trial_results, sim_trajectory = trial_with_switch_models(
            obj_id=obj_id,
            model=model,
            switch_model=switch_model,
            history_for_models=[history_for_model, history_for_switch_model],
            n_step=30,
            gui=False,
            website=cfg.website,
            all_joint=True,
            available_joints=available_links,
        )
        sim_trajectories += sim_trajectory
        link_names += [f"{obj_id}_{link}" for link in available_links]

        # Wandb table
        if obj_cat not in category_counts.keys():
            category_counts[obj_cat] = 0
        category_counts[obj_cat] += len(trial_results)
        for result in trial_results:
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
            wandb_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["count"] = category_counts[obj_cat]
            wandb_df.loc[obj_cat]["obj_cat"] = 0 if "train" in obj_cat else 1

        table = wandb.Table(dataframe=wandb_df.reset_index())
        run.log({f"simulation_metric_table": table})

    print(wandb_df)

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
