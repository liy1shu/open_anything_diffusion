import copy
import pickle as pkl

import numpy as np
import torch
import tqdm
from hydra import compose, initialize

from open_anything_diffusion.metrics.trajectory import flow_metrics

initialize(config_path="../../configs", version_base="1.3")
cfg = compose(config_name="eval_sim")

from open_anything_diffusion.models.flow_diffuser_hispndit import (
    FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
)
from open_anything_diffusion.models.modules.dit_models import PN2HisDiT
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder


class InferenceConfig:
    def __init__(self):
        self.batch_size = 1
        self.trajectory_len = 1
        self.mask_input_channel = False


inference_config = InferenceConfig()


class ModelConfig:
    def __init__(self):
        self.num_train_timesteps = 100


model_config = ModelConfig()

network = {
    "DiT": PN2HisDiT(
        history_embed_dim=128,
        in_channels=3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        # depth=8,
        # hidden_size=256,
        # num_heads=4,
        learn_sigma=True,
    ).cuda(),
    "History": HistoryEncoder(
        history_dim=128,
        history_len=1,
        batch_norm=True,
        transformer=False,
        repeat_dim=False,
    ).cuda(),
}
# ckpt_file = "/home/yishu/open_anything_diffusion/logs/train_trajectory_diffuser_hispndit/2024-05-25/02-00-54/checkpoints/epoch=399-step=331600-val_loss=0.00-weights-only.ckpt"
# ckpt_file = "/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_hispndit.ckpt"
ckpt_file = (
    "/home/yishu/open_anything_diffusion/pretrained/door_half_half_hispndit.ckpt"
)
model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
    network, inference_cfg=inference_config, model_cfg=model_config
).cuda()
model.load_from_ckpt(ckpt_file)
model.eval()


import json
import os


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"../../scripts/umpnet_data_split_new.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for train_or_test, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = f"{category}_{train_or_test}"

    else:
        with open(f"../../scripts/umpnet_object_list.json", "r") as f:
            data = json.load(f)
        for split in ["train-train", "train-test"]:
            # for split in ["train-test"]:
            for id in toy_dataset[split]:
                id_to_cat[id] = split
    return id_to_cat


pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
id_to_cat = load_obj_id_to_category(None)
with open(
    "/home/yishu/open_anything_diffusion/scripts/movable_links_fullset_000.json", "r"
) as f:
    obj_dict = json.load(f)


from open_anything_diffusion.models.flow_diffuser_hispndit import (
    FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
)
from open_anything_diffusion.models.modules.dit_models import PN2HisDiT
from open_anything_diffusion.models.modules.history_encoder import HistoryEncoder

ckpt_file = (
    "/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_hispndit.ckpt"
)

network = {
    "DiT": PN2HisDiT(
        history_embed_dim=128,
        in_channels=3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        learn_sigma=True,
    ).cuda(),
    "History": HistoryEncoder(
        history_dim=128,
        history_len=1,
        batch_norm=True,
        transformer=False,
        repeat_dim=False,
    ).cuda(),
}


class InferenceConfig:
    def __init__(self):
        self.batch_size = 1
        self.trajectory_len = 1
        self.mask_input_channel = False


inference_config = InferenceConfig()


class ModelConfig:
    def __init__(self):
        self.num_train_timesteps = 100


model_config = ModelConfig()

model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
    network, inference_cfg=inference_config, model_cfg=model_config
)
model.load_from_ckpt(ckpt_file)
model.eval()
model.cuda()


import pybullet as p
from rpad.partnet_mobility_utils.data import PMObject
from tqdm import tqdm

from open_anything_diffusion.metrics.trajectory import flow_metrics
from open_anything_diffusion.simulations.suction import GTFlowModel, PMSuctionSim

all_categories = list(set(list(id_to_cat.values())))
all_categories_name = list(set([name.split("_")[0] for name in all_categories]))
print(all_categories_name)

# test_categories = ['Door_test', 'Door_train']

trial_cnts = 20
multimodal = {}
can_be_correct = {}
counts = {}
mags = {}
cosines = {}
rmse_records = {}
for name in all_categories:
    multimodal[name] = 0
    can_be_correct[name] = 0
    counts[name] = 0

    mags[name] = []
    cosines[name] = []
    rmse_records[name] = []

might_fail_trials = []
for obj_id, joint_ids in tqdm(obj_dict.items()):
    if "train" not in id_to_cat[obj_id]:
        continue
    # print(obj_id, joint_ids)
    for joint_id in joint_ids:
        print(obj_id, id_to_cat[obj_id], counts.keys())
        counts[id_to_cat[obj_id]] += 1
        raw_data = PMObject(os.path.join(pm_dir, obj_id))
        env = PMSuctionSim(obj_id, pm_dir, gui=False)
        gt_model = GTFlowModel(raw_data, env)
        env.reset()

        target_link = joint_id
        info = p.getJointInfo(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[target_link],
            env.render_env.client_id,
        )
        init_angle, target_angle = info[8], info[9]

        env.set_joint_state(
            target_link,
            init_angle + np.linspace(0, 1, 20)[0] * (target_angle - init_angle),
        )
        # env.set_joint_state(target_link, target_angle)
        pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        gt_flow = gt_model(pc_obs)
        nonzero_gt_flowixs = pc_seg == env.render_env.link_name_to_index[target_link]
        gt_flow_nz = gt_flow[nonzero_gt_flowixs]

        has_pos = False
        has_neg = False
        rmses = []
        cos_dists = []
        mag_errors = []

        for i in tqdm(range(trial_cnts)):
            with torch.no_grad():
                # pred_flow = model(copy.deepcopy(pc_obs))[:, 0, :]
                pred_flow = model(
                    copy.deepcopy(pc_obs),
                    history_pcd=None,
                    history_flow=None,
                )[:, 0, :]

            pred_flow_nz = pred_flow[nonzero_gt_flowixs]

            rmse, cos_dist, mag_error = flow_metrics(pred_flow_nz, gt_flow_nz)
            rmses.append(rmse)
            cos_dists.append(cos_dist)
            mag_errors.append(mag_error)
            # if cos_dist > 0.8 and mag_error < 0.3:
            if rmse < 0.2:
                has_pos = True
            if rmse > 0.6:
                has_neg = True

        can_be_correct[id_to_cat[obj_id]] += has_pos
        multimodal[id_to_cat[obj_id]] += has_pos and has_neg
        mags[id_to_cat[obj_id]].append(mag_errors)
        cosines[id_to_cat[obj_id]].append(cos_dists)
        rmse_records[id_to_cat[obj_id]].append(rmses)

        all_results_dict = {
            "multimodal": multimodal,
            "mags": mags,
            "cosines": cosines,
            "rmses": rmse_records,
        }
        with open("./multimodal_dict.pkl", "wb") as f:
            pkl.dump(all_results_dict, f)

print(multimodal)
breakpoint()
