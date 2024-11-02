# Simulation results for diffusion models

import json

from diffuser import TrainingConfig, TrajDiffuser, TrajDiffuserSimWrapper

from open_anything_diffusion.simulations.simulation import trial_with_diffuser


def load_obj_and_link():
    with open("../../../../scripts/umpnet_object_list.json", "r") as f:
        object_link_json = json.load(f)
    return object_link_json


object_to_link = load_obj_and_link()

config = TrainingConfig()
diffuser = TrajDiffuser(config, train_batch_num=1)
diffuser.load_model("./half_diffusion_fullset_ckpt_backup.pth")

diffuser_simulator = TrajDiffuserSimWrapper(diffuser)

success_rate = 0
norm_dist = 0
count = 0
import tqdm

object_ids = [
    "8867",
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
for obj_id in tqdm.tqdm(object_ids):
    if obj_id not in object_to_link.keys():
        continue
    available_links = object_to_link[obj_id]
    trial_figs, trial_results = trial_with_diffuser(
        obj_id=obj_id,
        model=diffuser_simulator,
        n_step=30,
        gui=False,
        website=True,
        all_joint=True,
        # available_joints=available_links,
    )

    for result in trial_results:
        if not result.contact:
            continue
        success_rate += result.success
        norm_dist += result.metric
        count += 1
    # all_trial_results.append(trial_results)
    # all_trial_figs.append(trial_figs)

print(success_rate / count, norm_dist / count)

breakpoint()
