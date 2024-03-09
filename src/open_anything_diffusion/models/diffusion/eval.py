from diffuser import TrainingConfig, TrajDiffuser

config = TrainingConfig()
diffuser = TrajDiffuser(config, train_batch_num=1)
diffuser.load_model("./door_diffusion_ckpt_valbest.pth")

# diffuser.load_model('/home/yishu/open_anything_diffusion/logs/train_trajectory/2023-08-31/16-13-10/checkpoints/epoch=399-step=314400.ckpt')

from open_anything_diffusion.datasets.flow_trajectory import FlowTrajectoryDataModule

datamodule = FlowTrajectoryDataModule(
    root="/home/yishu/datasets/partnet-mobility",
    batch_size=1,
    num_workers=30,
    n_proc=2,
    seed=42,
    trajectory_len=config.traj_len,  # Only used when training trajectory model
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
        "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
        "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
    },
)

train_dataloader = datamodule.train_dataloader()
train_val_dataloader = datamodule.train_val_dataloader()
unseen_dataloader = datamodule.unseen_dataloader()

trial_time = 50
all_directions = []
import tqdm

for dataloader in [train_val_dataloader, unseen_dataloader]:
    total_metrics = {
        "best_rmse": 0,
        "best_cosine": 0,
        "best_mag": 0,
        "best_flow_loss": 0,
        "multimodal_ratio": 0,
        "pos_rate": 0,
        "neg_rate": 0,
    }
    multimodal_cnt = 0
    total_sample_cnt = 0
    for sample in tqdm.tqdm(dataloader):
        total_sample_cnt += 1
        # best_metric = {
        #     'rmse': tensor(10.0, device='cuda:0'),
        #     'cos_dist': tensor(-10.0, device='cuda:0'),
        #     'mag_error': tensor(10.0, device='cuda:0'),
        #     'flow_loss': tensor(10.0, device='cuda:0'),
        #     'multimodal': 0,
        # }
        # multimodal = False
        # for i in range(trial_time):
        #     # metric = diffuser.predict([sample], vis=False)
        #     metric = diffuser.predict_wta([sample], trial_times=trial_time)
        #     all_directions += metric['all_directions']
        #     # print(metric)
        #     if i!=0 and metric['cos_dist'] * best_metric['cos_dist'] < 0:  # predicts different directions
        #         multimodal = True
        #     # if metric['cos_dist'] > best_metric['cos_dist']:
        #     if metric['flow_loss'] < best_metric['flow_loss']:
        #         for metric_type in best_metric.keys():
        #             best_metric[metric_type] = metric[metric_type]

        # multimodal = False
        # for i in range(trial_time):
        # metric = diffuser.predict([sample], vis=False)
        best_metric = diffuser.predict_wta([sample], trial_times=trial_time)
        all_directions += best_metric["all_directions"]
        multimodal = best_metric["multimodal_ratio"]
        # print(metric)
        # if i!=0 and metric['cos_dist'] * best_metric['cos_dist'] < 0:  # predicts different directions
        #     multimodal = True
        # # if metric['cos_dist'] > best_metric['cos_dist']:
        # if metric['flow_loss'] < best_metric['flow_loss']:
        #     for metric_type in best_metric.keys():
        #         best_metric[metric_type] = metric[metric_type]

        # print(multimodal, best_metric["best_cosine"])
        for metric_type in total_metrics.keys():
            total_metrics[metric_type] += best_metric[metric_type]  # .item()

        multimodal_cnt += multimodal

    for metric_type in total_metrics.keys():
        # total_metrics[metric_type] /= len(train_val_dataloader)
        total_metrics[metric_type] /= total_sample_cnt

    print(total_metrics)
    print(multimodal_cnt)


# Scatter plot
ys = [d.item() for d in all_directions]
xs = [
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

all_xs = []
for x in xs:
    all_xs += [x] * trial_time
# breakpoint()
colors = sorted(["red"]) * trial_time * (len(xs) - 6) + ["blue"] * trial_time * 6
# colors = sorted(["red", "blue", "yellow"] * trial_time) * len(all_xs)
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.axhline(y=0)
plt.scatter(all_xs, ys, s=5, c=colors[: len(ys)])
plt.xticks(rotation=90)
plt.savefig("./door_cos_stats.jpeg")
