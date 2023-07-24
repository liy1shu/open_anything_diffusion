# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
from python_ml_project_template.simulations.pm_raw import PMRawData
from python_ml_project_template.simulations.suction import (  # compute_flow,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
)


def trial_flow(obj_id="41083", gui=False):
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMRawData(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = GTFlowModel(raw_data, env)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    run_trial(env, raw_data, joint.name, model, n_flow_steps=30)


# TODO: trial with groundtruth trajectories
def trial_gt_trajectory(obj_id="41083", gui=False):
    pass


# TODO: trial with model predicted trajectories
def trial_with_prediction(obj_id="41083", gui=False):
    pass


if __name__ == "__main__":
    trial_flow(obj_id="41083", gui=False)
