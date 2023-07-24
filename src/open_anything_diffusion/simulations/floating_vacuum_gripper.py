from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

# from part_embedding.envs.debug_viz import draw_pose

__ASSETS_CHUNK = "third_party/ravens/ravens/environments/assets"
ASSETS_DIR = Path(__file__).parent.parent.parent.parent / __ASSETS_CHUNK
print(ASSETS_DIR)
SUCTION_BASE_URDF = str(ASSETS_DIR / "ur5/suction/suction-base.urdf")
SUCTION_HEAD_URDF = str(ASSETS_DIR / "ur5/suction/suction-head.urdf")


class FloatingSuctionGripper:
    def __init__(self, obj_id: int, client_id):
        self.client_id = client_id
        self.obj_id = obj_id
        # TODO: update this to support multiple objects.

        # Load in the base, which can be thought of as the robot's wrist.
        # This is a visual mesh, and has no collision.
        # TODO: Decide if we can swap this out with the sawyer arm or something.
        base_pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.base_id = p.loadURDF(
            SUCTION_BASE_URDF, *base_pose, physicsClientId=self.client_id
        )

        # Load suction tip model (visual and collision).
        tip_pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body_id = p.loadURDF(
            SUCTION_HEAD_URDF, *tip_pose, physicsClientId=self.client_id
        )
        # TODO: FIX THIS HACK THAT MAKES THE GRIPPER SUPER HEAVY TO MAKE
        # MOTION STABLE. It's carrying the object forward w/ inertia, essentialy.
        p.changeDynamics(bodyUniqueId=self.body_id, linkIndex=0, mass=10)

        # We need to create a compliant constraint between the tip and the base.
        # This way when it makes contact, it can bounce a bit. Also, contact
        # won't necessarily be flush.
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08),
            physicsClientId=self.client_id,
        )
        p.changeConstraint(
            constraint_id, maxForce=10000, physicsClientId=self.client_id
        )

        # Keep track of whether or not the gripper is in activated.
        self.activated = False
        self.contact_const = None
        self.contact_link_index = None

    def set_pose(self, pos, ori):
        p.resetBasePositionAndOrientation(self.body_id, pos, ori, self.client_id)
        p.resetBasePositionAndOrientation(self.base_id, pos, ori, self.client_id)

    def set_velocity(self, lin_vel, ang_vel):
        p.resetBaseVelocity(self.body_id, lin_vel, ang_vel, self.client_id)
        p.resetBaseVelocity(self.base_id, lin_vel, ang_vel, self.client_id)

    def detect_contact(self):
        # Get all the contact points that the gripper tip (bodyA) is experiencing.
        # Restrict contact detection to the object of interest.
        points = p.getContactPoints(bodyA=self.body_id, bodyB=self.obj_id, linkIndexA=0)
        return len(points) != 0

    def apply_force(self, force):
        # TODO: FIGURE OUT why applying force on a constrained object doesn't
        # seem to work right.
        base_link_pos, base_link_ori, _, _, _, _ = p.getLinkState(
            bodyUniqueId=self.base_id,
            linkIndex=0,
            computeLinkVelocity=0,
            physicsClientId=self.client_id,
        )
        p.applyExternalForce(
            self.base_id,
            linkIndex=0,
            forceObj=force,
            posObj=base_link_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )

    def activate(self):
        if not self.activated:
            points = p.getContactPoints(
                bodyA=self.body_id, bodyB=self.obj_id, linkIndexA=0
            )
            if points is not None:
                # We'll choose the first point as the contact.
                point = points[0]
                contact_pos_on_A, contact_pos_on_B = point[5], point[6]
                obj_id, contact_link = point[2], point[4]

                # Describe the contact point in the TIP FRAME.
                base_link_pos, base_link_ori, _, _, _, _ = p.getLinkState(
                    bodyUniqueId=self.body_id,
                    linkIndex=0,
                    computeLinkVelocity=0,
                    physicsClientId=self.client_id,
                )
                T_world_tip = base_link_pos, base_link_ori
                T_tip_world = p.invertTransform(*T_world_tip)
                T_world_contact = contact_pos_on_A, base_link_ori
                T_tip_contact = p.multiplyTransforms(*T_tip_world, *T_world_contact)

                # Describe the contact point in the OBJ FRAME. Note that we use the
                # orientation of the gripper
                objcom_link_pos, objcom_link_ori, _, _, _, _ = p.getLinkState(
                    bodyUniqueId=obj_id,
                    linkIndex=contact_link,
                    computeLinkVelocity=0,
                    physicsClientId=self.client_id,
                )
                T_world_objcom = objcom_link_pos, objcom_link_ori
                T_objcom_world = p.invertTransform(*T_world_objcom)
                T_obj_contact = p.multiplyTransforms(*T_objcom_world, *T_world_contact)

                # Short names so we can debug later (parent frame; child frame),
                pfp, pfo = T_tip_contact
                cfp, cfo = T_obj_contact

                # debug_pose(*T_tip_contact, self.body_id, 0, self.client_id)
                # debug_pose(*T_obj_contact, obj_id, contact_link, self.client_id)

                self.contact_const = p.createConstraint(
                    parentBodyUniqueId=self.body_id,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=pfp,
                    parentFrameOrientation=pfo,
                    childFramePosition=cfp,
                    childFrameOrientation=cfo,
                )

                # debug_constraint(
                #     p.getConstraintInfo(self.contact_const, self.client_id)
                # )

                self.activated = True
                self.contact_link_index = contact_link

    def release(self):
        if self.contact_const:
            p.removeConstraint(self.contact_const)
        self.activated = False
        self.contact_link_index = None


def debug_pose(pos, ori, parent_id=None, parent_link=None, client_id=None):
    T_parent_contact = np.eye(4)
    T_parent_contact[:3, :3] = R.from_quat(ori).as_matrix()
    T_parent_contact[:3, 3] = pos
    # draw_pose(
    #     T=T_parent_contact,
    #     parent_id=parent_id,
    #     parent_link=parent_link,
    #     client_id=client_id,
    # )


def debug_contact(points):
    for i, point in enumerate(points):
        (
            contactFlag,
            bodyUniqueIdA,
            bodyUniqueIdB,
            linkIndexA,
            linkIndexB,
            positionOnA,
            positionOnB,
            contactNormalOnB,
            contactDistance,
            normalForce,
            lateralFriction1,
            lateralFrictionDir1,
            lateralFriction2,
            lateralFrictionDir2,
        ) = point

        print(f"----- Contact {i} -----")
        print(f"Body A: {p.getBodyInfo(bodyUniqueIdA)[1].decode('utf-8')}")
        print(f"Body B: {p.getBodyInfo(bodyUniqueIdB)[1].decode('utf-8')}")
        print(f"Link Index A: {linkIndexA}")
        print(f"Link Index B: {linkIndexB}")
        print(f"Position on A: {positionOnA}")
        print(f"Position on B: {positionOnB}")


def debug_constraint(constraint_info):
    (
        parentBodyUniqueId,
        parentJointIndex,
        childBodyUniqueId,
        childLinkIndex,
        constraintType,
        jointAxis,
        jointPivotInParent,
        jointPivotInChild,
        jointFrameOrientationParent,
        jointFrameOrientationChild,
        maxAppliedForce,
        gearRatio,
        gearAuxLink,
        relativePositionTarget,
        erp,
    ) = constraint_info
    print(f"----- Constraint -----")
    print(f"Body Parent: {p.getBodyInfo(parentBodyUniqueId)[1].decode('utf-8')}")
    print(f"Body Child: {p.getBodyInfo(childBodyUniqueId)[1].decode('utf-8')}")
    print(f"Link Index Parent: {parentJointIndex}")
    print(f"Link Index Child: {childLinkIndex}")
    print(f"Max Applied Force: {maxAppliedForce}")
