# Only run this test if the PARTNET_MOBILITY_DIR environment variable is set
import os

import pytest

from open_anything_diffusion.simulations.simulation import trial_flow

if "PARTNET_MOBILITY_DIR" not in os.environ:
    pytest.skip("PARTNET_MOBILITY_DIR not set, skipping", allow_module_level=True)


def test_gt_agent_trial():
    result = trial_flow(
        obj_id="41083",
        gui=False,
        pm_dir=os.path.expanduser(os.environ["PARTNET_MOBILITY_DIR"]),
    )
