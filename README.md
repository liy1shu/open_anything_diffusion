# open_anything_diffusion

This repo uses diffusion to handle ambiguous open action planning.

## Run Full-set Experiments

### Train

STEP-1: Specify the Config files:
1) configs/train.yaml: 

    Choose dataset (single flow or trajectory) and model (original flowbot or diffuser)
    - dataset: trajectory / flowbot
    - model: artflownet / diffuser

2) configs/training/{}_{}.yaml

    Change the corresponding detailed configs for the training process: learning_rate, batch_size, warmup_steps, etc.

3) configs/model/{}.yaml

    Change the detailed configs for the model (Only needed for diffusion)

    - num_train_timesteps: diffusion timestep

STEP-2: Run training script
```
python scripts/train.py
```

### Eval (Currently only for flowbot)

Basically the same procedure with training, only the config files are eval.yaml (metric evaluation), eval_sim.yaml (simulation evaluation).

Need to specify:
checkpoint/run_id: the run_id in wandb
wandb/group: the group name in wandb

```
python scripts/eval.py
python scripts/eval_sim.py
```

## Run Small Dataset Experiment

Need to change scripts/train.py for training and scripts/eval(_sim).py for evaluation:

When creating dataset, specify the arguments `special_req` and `toy_dataset`.

1) special_req: 

- "half-half"(Half of data fully closed, half of data randomly opened )
- "fully-closed"(All of data fully closed)

2) toy_dataset: a dict to specify a small dataset
- id: the name for the toy dataset
- train-train: the ids for the training set
- train-test: the ids for the validation set
- test: the ids for the test set

An Example:
```
# Create FlowBot dataset
datamodule = data_module_class[cfg.dataset.name](
    root=cfg.dataset.data_dir,
    batch_size=cfg.training.batch_size,
    num_workers=cfg.resources.num_workers,
    n_proc=cfg.resources.n_proc_per_worker,
    seed=cfg.seed,
    trajectory_len=trajectory_len, 
    special_req="half-half"
    # only for toy training
    toy_dataset = {
        "id": "door-1",
        "train-train": ["8994", "9035"],
        "train-test": ["8994", "9035"],
        "test": ["8867"],
    }
)
```

Then run train and eval exactly like before.

## Run Diffusion

Currently For diffusion, most experiments are run with scripts under `src/open_anything_diffusion/models/diffusion`. (Although the above set of pipeline is also complete, I suggest currently run diffusion under this directory and with following commands)

Train: Under `src/open_anything_diffusion/models/diffusion/`
```
python diffuser.py
```
Eval: Under `src/open_anything_diffusion/models/diffusion/`
```
python eval.py
```

We can also use `inference.ipynb` to see the visualization results