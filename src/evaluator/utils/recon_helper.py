import torch
from .rotation_conversions import *
from ..data.dnd_config import *


def recover_from_ric(data, is_motion=False):
    # data: B, F, 6+3*69
    # r_rot_quat: [F, 3, 3], r_pos: [F, 3]
    #r_rot_mat, r_pos = recover_root_rot_pos(data)
    r_rot_mat_inv = rotation_6d_to_matrix(data[..., :6])
    #print('r_rot_mat_', r_rot_mat_[10])
    
    #r_rot_quat = matrix_to_quaternion(rotation_6d_to_matrix(data[..., :6]))
    #r_pos: B, F, 1, 3
    if is_motion is True:
        r_pos = torch.zeros((data[..., 6:9].shape)).to(data.device)
        r_pos[..., 1:, :] = torch.cumsum(data[..., 6:9], dim=-2)[..., :-1, :]
        r_pos = r_pos.unsqueeze(-2)
    else:
        r_pos = data[..., 6:6+3].unsqueeze(-2)
    positions = data[..., 9:]
    positions = positions.reshape(*data.shape[:-1], -1, 3)
    
    '''Add Y-axis rotation to local joints'''
    # B, F, 1, 3, 3    B, F, 69, 3, 1
    positions = torch.matmul(r_rot_mat_inv.unsqueeze(-3), positions.unsqueeze(-1))[..., 0]
    
    '''Add root XZ to joints'''
    positions += r_pos

    '''Concate root and joints'''
    positions = torch.cat([r_pos, positions], dim=-2)
    
    return positions

def compose_whole_body_pose(positions):
    # recon root pos
    # positions_out: B, T, 69, 3
    # positions_out = torch.cat([root_pos, body_pos, left_hand_pos_wo_wrist, right_hand_pos_wo_wrist], dim=-2)
    root_pos = positions[..., :1, :]
    body_pos = positions[..., 1:-42, :]
    body_pos += root_pos
    
    left_hand = positions[..., -42:-21, :] + body_pos[..., body_wrist_l-1:body_wrist_l, :]
    right_hand = positions[..., -21:, :] + body_pos[..., body_wrist_r-1:body_wrist_r, :]
    
    '''Combine root and joints'''
    positions_whole_body = torch.zeros_like(positions).to(positions.device)
    positions_whole_body[..., dnd_body_idxs[:1], :] = root_pos
    positions_whole_body[..., dnd_body_idxs[1:], :] = body_pos
    positions_whole_body[..., dnd_left_hand_idxs[1:], :] = left_hand
    positions_whole_body[..., dnd_right_hand_idxs[1:], :] = right_hand
    
    return positions_whole_body
'''
def skeleton_forward_kinematics(skeleton, pose_repr):
    
    B = pose_repr.shape[0]
    F = pose_repr.shape[1]
    
    # B, F, 3 + 54*6
    global_pos = pose_repr[..., :3]
    rot_loc = pose_repr[..., 3:].reshape(B, F, -1, 6)
    # B, F, 54, 6 -> B, F, 54, 3, 3
    rot_loc = rotation_6d_to_matrix(rot_loc)

    #skeleton = Skeleton(joint_init, dtype=rot_loc.dtype, device=rot_loc.device)
    # rot_loc: B, F, 54, 3, 3 -> B*F, 54, 3, 3
    # out_pose: B*F, 69, 3
    out_pose = skeleton.forward_pose(rot_loc.reshape(B*F, -1, 3, 3))
    # mm to m
    out_pose = out_pose.reshape(B, F, -1, 3)
    
    # mm to m
    out_pose = out_pose * 0.001 + global_pos
    
    return out_pose'''