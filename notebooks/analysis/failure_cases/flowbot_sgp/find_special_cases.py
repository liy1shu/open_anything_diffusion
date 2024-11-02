import json

import numpy as np
import torch
from tqdm import tqdm

from open_anything_diffusion.simulations.simulation import *

np.random.seed(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)
from hydra import compose, initialize

initialize(config_path="../../../../configs", version_base="1.3")
cfg = compose(config_name="eval_sim_switch")
# cfg = compose(config_name="eval_sim")


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"../../../../scripts/umpnet_data_split_new.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for train_or_test, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = f"{category}_{train_or_test}"

    else:
        with open(f"../../../../scripts/umpnet_object_list.json", "r") as f:
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


# special_type = 'occlusion'

occluded_txt = "./occluded_ids_MD.txt"
ckpt_file = (
    "/home/yishu/open_anything_diffusion/pretrained/fullset_half_half_flowbot.ckpt"
)

import rpad.pyg.nets.pointnet2 as pnp

network = pnp.PN2Dense(
    in_channels=0,
    out_channels=3,
    p=pnp.PN2DenseParams(),
)

# Load the network weights.
ckpt = torch.load(ckpt_file)
network.load_state_dict(
    {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
)
network.eval()


all_categories = list(set(list(id_to_cat.values())))
all_categories_name = list(set([name.split("_")[0] for name in all_categories]))
print(all_categories_name)

# # might_occlude_names = ['WashingMachine', 'Door', 'Safe', 'Dishwasher', 'Refrigerator', 'Microwave', 'Oven']
might_occlude_names = ["Microwave"]
might_occludes = [name + "_test" for name in might_occlude_names]

# Statistics: switch grasp point & success rate
total_sample_counts = 0
switch_grasp_counts = []  # Record the switch grasp time for each sample
step_counts = []  # Record the step count for each sample
switch_grasp_per_step = np.zeros(31)
success_rate_per_step = np.zeros(31)

might_be_occluded_trials = []
for obj_id, joint_ids in tqdm(obj_dict.items()):
    # if id_to_cat[obj_id] not in might_occludes:   # Filter objects
    #     continue
    # print(obj_id, joint_ids)
    for joint_id in joint_ids:
        raw_data = PMObject(os.path.join(pm_dir, obj_id))
        # available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        #     "slider"
        # )
        # available_joints = [joint.name for joint in available_joints]
        # target_link = available_joints[joint_id]
        # print(target_link)
        target_link = joint_id

        # FlowBot
        trial_figs, trial_results, all_signals = trial_with_prediction(
            obj_id=obj_id,
            network=network,
            n_step=30,
            gui=False,
            all_joint=False,
            available_joints=[target_link],
            website=True,
            sgp=True,
            analysis=True,
        )
        # print(trial_results, all_signals)
        try:
            (
                sim_trajectory,
                update_history_signals,
                cc_cnts,
                sgp_signals,
                visual_all_points,
                visual_link_ixs,
                visual_grasp_points_idx,
                visual_grasp_points,
                visual_flows,
            ) = all_signals[0]
            total_sample_counts += 1
            is_success = sim_trajectory[-1] >= 0.9
            step = len(visual_all_points)
            step_counts.append(step)
            switch_grasp_counts.append(np.sum(sgp_signals))  # Include the first grasp

            success_rate_per_step[step:] += is_success
            switch_grasp_per_step[: len(sgp_signals)] += np.array(sgp_signals)

            route_str = " -> ".join(f"{num:.4f}" for num in sim_trajectory)
            if sim_trajectory[-1] >= 1:  # Has already opened!
                continue
            for step in range(1, len(sim_trajectory)):
                if sim_trajectory[step] < sim_trajectory[step - 1] - 1e-4:
                    might_be_occluded_trials.append(
                        [trial_figs, trial_results, all_signals[0]]
                    )
                    with open(occluded_txt, "a") as f:
                        f.write(f"Id: {obj_id}, link: {joint_id}:\n")
                        f.write(f"{route_str}\n\n")
                    break
        except:
            continue

still_running_per_step = total_sample_counts - success_rate_per_step
success_rate_per_step /= total_sample_counts
mean_switch_grasp_counts = np.mean(switch_grasp_counts)
mean_step_counts = np.mean(step_counts)
print(f"Mean grasp counts: {mean_switch_grasp_counts}")
print(f"Mean step counts: {mean_step_counts}")
for i in range(len(switch_grasp_per_step)):
    if still_running_per_step[i] != 0:
        switch_grasp_per_step[i] /= still_running_per_step[i]

# Make plots!!!
import matplotlib.pyplot as plt

# Make plots!!!
# 0 - Success rate inc curve
success_rate_per_step_inc = [0]
success_rate_per_step_inc += [
    success_rate_per_step[i] - success_rate_per_step[i - 1]
    for i in range(1, len(success_rate_per_step))
]

x = list(range(0, len(success_rate_per_step_inc)))
plt.plot(x, success_rate_per_step_inc, marker="o")
plt.ylim(-0.01, 0.25)
plt.xlim(-2, 32)
plt.xlabel("Step Id")
plt.ylabel("Success Rate Increase")
plt.title("Success Rate Inc per Step (FlowBotHD w/ SGP)")
plt.grid(True)
plt.savefig("./success_rate_inc_curve.jpg")
plt.clf()


# 1 - Success rate curve
x = list(range(0, len(success_rate_per_step)))
plt.plot(x, success_rate_per_step, marker="o")
plt.ylim(-0.05, 1.05)
plt.xlim(-2, 32)
plt.xlabel("Step Id")
plt.ylabel("Success Rate")
plt.title("Success Rate per Step (FlowBot w/ SGP)")
plt.grid(True)

plt.savefig("./success_rate_curve.jpg")
plt.clf()

# 2 - Switch grasp curve
x = list(range(1, len(switch_grasp_per_step)))
plt.plot(x, switch_grasp_per_step[:-1], marker="o", c="orange")
plt.ylim(-0.05, 1.05)
plt.xlabel("Step Id")
plt.ylabel("Switch Grasp Rate")
plt.title("Switch Grasp Rate per Step (FlowBot)")
plt.grid(True)

plt.savefig("./switch_grasp_curve.jpg")
plt.clf()

# 3 - Switch grasp count hist
plt.hist(switch_grasp_counts, bins=20, color="blue", edgecolor="black", density=True)
plt.xlabel("Switch Counts")
plt.ylabel("Frequency")
plt.title("Distribution of Switch Grasp Points (FlowBot w/ SGP)")
plt.savefig("./switch_grasp_hist.jpg")
plt.clf()

breakpoint()
