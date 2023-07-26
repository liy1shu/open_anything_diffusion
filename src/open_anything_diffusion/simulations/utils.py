import functools
import os
import sys
from contextlib import contextmanager

import pybullet as p


def get_obj_z_offset(object_id, sim_id, starting_min=0.0):
    bboxes = [p.getAABB(object_id, physicsClientId=sim_id)]  # Add the base one.
    for i in range(p.getNumJoints(object_id, physicsClientId=sim_id)):
        aabb = p.getAABB(object_id, i, physicsClientId=sim_id)  # Add the links.
        bboxes.append(aabb)
    minz = functools.reduce(lambda a, b: min(a, b[0][2]), bboxes, starting_min)
    return minz


def get_obj_bbox_xy(object_id, sim_id):
    bboxes = [p.getAABB(object_id, physicsClientId=sim_id)]  # Add the base one.
    for i in range(p.getNumJoints(object_id, physicsClientId=sim_id)):
        aabb = p.getAABB(object_id, i, physicsClientId=sim_id)  # Add the links.
        bboxes.append(aabb)
    # xmin = np.array(bboxes)[:, :, 0].min()
    xmin = functools.reduce(lambda a, b: min(a, b[0][0]), bboxes, 0.0)
    # xmax = np.array(bboxes)[:, :, 0].max()
    xmax = functools.reduce(lambda a, b: max(a, b[1][0]), bboxes, 0.0)
    # ymin = np.array(bboxes)[:, :, 1].min()
    ymin = functools.reduce(lambda a, b: min(a, b[0][1]), bboxes, 0.0)
    # ymax = np.array(bboxes)[:, :, 1].max()
    ymax = functools.reduce(lambda a, b: max(a, b[1][1]), bboxes, 0.0)
    return [[xmin, xmax], [ymin, ymax]]


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
