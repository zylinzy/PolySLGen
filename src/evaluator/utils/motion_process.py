from typing import Dict
import numpy as np
from data.dnd_config import *
from .quaternion import *

def get_g_rot(motion):
    
    root_pos_init = motion[0]
    # '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = DND_JOINT_NAMES.index('RightUpLeg'), DND_JOINT_NAMES.index('LeftUpLeg'), DND_JOINT_NAMES.index('RightShoulder'), DND_JOINT_NAMES.index('LeftShoulder') #18, 13, 9, 5
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init
    
    return root_quat_init

def process_motion( motion):
    # breakpoint()
    #  Put on floor
    # T, J, 3
    floor_height = motion.min(axis=0).min(axis=0)[1]
    motion[:, :, 1] -= floor_height

    #  '''XZ at origin'''
    root_pos_init = motion[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    motion = motion - root_pose_init_xz

    # '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = DND_JOINT_NAMES.index('RightUpLeg'), DND_JOINT_NAMES.index('LeftUpLeg'), DND_JOINT_NAMES.index('RightShoulder'), DND_JOINT_NAMES.index('LeftShoulder') #18, 13, 9, 5
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init

    motion_b = motion.copy()

    motion = qrot_np(root_quat_init, motion)

    # all joints root relative
    motion[:, 1:, :] = motion[:, 1:, :] - motion[:, :1, :] 

    # hands relative to wrist

    motion[:, DND_JOINT_NAMES.index('LeftHandThumb1'):DND_JOINT_NAMES.index('LeftHand_end'), :] = motion[:, DND_JOINT_NAMES.index('LeftHandThumb1'):DND_JOINT_NAMES.index('LeftHand_end'), :] - motion[:, [DND_JOINT_NAMES.index('LeftHand')], :]
    motion[:, DND_JOINT_NAMES.index('RightHandThumb1'):DND_JOINT_NAMES.index('RightHand_end'), :] = motion[:, DND_JOINT_NAMES.index('RightHandThumb1'):DND_JOINT_NAMES.index('RightHand_end'), :] - motion[:, [DND_JOINT_NAMES.index('RightHand')], :]
    #motion[:, 43:, :] = motion[:, 43:, :] - motion[:, [11], :]
    # motion[:, 23:, :] = motion[:, 23:, :] * 10
    # motion = motion * 3 # all equal scale

    # motion = motion.reshape(-1, 63 * 3)
    # global velocity
    motion[:-1, :1] = motion[1:, :1] - motion[:-1, :1]
    motion = motion[:-1]
    # motion: T, 69, 3 -- > T, 69*3
    motion = motion.reshape(motion.shape[0], -1)
    # root_quat_init: T, 4
    root_quat_init = root_quat_init[:-1, 0]
    root_quat_init = quaternion_to_cont6d_np(root_quat_init)
    return np.concatenate((root_quat_init, motion), axis=-1)


def decompose(datas):
    g_rot = datas[..., :6]
    motion = datas[..., 6:].reshape(*datas.shape[:-1], -1, 3)
    return g_rot, motion
