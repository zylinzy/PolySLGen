from typing import Dict
import numpy as np
import torch
from .rotation_conversions import *

class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = torch.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])
    
import torch.nn as nn
class Skeleton(nn.Module):
    def __init__(self, joints_init, root_name='Hips'):
        super().__init__()
        
        self.joints: Dict[str, BvhJoint] = joints_init
        self.joints_wo_end = []
        self.root = self.joints[root_name]
        #self.device = device
        #self.dtype = dtype
        
        self.joints_offset = nn.ParameterDict()
        for k, v in self.joints.items():
            self.joints_offset[k] = nn.Parameter(self.joints[k].offset, requires_grad=False)
            if '_end' not in k:
                self.joints_wo_end += [k]
        
    def _recursive_apply_frame(self, joint, frame_pose, p, M_parent, p_parent):

        if '_end' in joint.name:
            joint_index = list(self.joints.values()).index(joint)
            p[..., joint_index, :] = p_parent + torch.matmul(M_parent, self.joints_offset[joint.name].to(dtype=M_parent.dtype))
            return 0
        
        joint_index_rot = self.joints_wo_end.index(joint.name)   
        M_rotation = frame_pose[..., joint_index_rot, :, :]
        
        M = torch.matmul(M_parent, M_rotation)
        position = p_parent + torch.matmul(M_parent, self.joints_offset[joint.name].to(dtype=M_parent.dtype))

        joint_index = list(self.joints.values()).index(joint)
        p[..., joint_index, :] = position
        
        for c in joint.children:
            self._recursive_apply_frame(c, frame_pose, p, M, position)

        return 0

    def forward_pose(self, frame_rot):
        #frame_rot: B, F, 54, 3, 3, or BF, 54, 3, 3
        #B = frame_rot.shape[0]
        p = torch.empty((*frame_rot.shape[:-3], len(self.joints), 3)).to(dtype=frame_rot.dtype, device=frame_rot.device)
        M_parent = torch.zeros((*frame_rot.shape[:-3], 3, 3)).to(dtype=frame_rot.dtype, device=frame_rot.device)
        M_parent[..., 0, 0] = 1
        M_parent[..., 1, 1] = 1
        M_parent[..., 2, 2] = 1
        
        self._recursive_apply_frame(self.root, frame_rot, p, M_parent, torch.zeros((*frame_rot.shape[:-3], 3)).to(dtype=frame_rot.dtype, device=frame_rot.device))
        
        return p
    
    def _recursive_apply_frame_rot(self, joint, frame_pose, r, M_parent, p_parent):

        if '_end' in joint.name:
            return 0
        
        joint_index_rot = self.joints_wo_end.index(joint.name)   
        M_rotation = frame_pose[..., joint_index_rot, :, :]
        
        M = torch.matmul(M_parent, M_rotation)
        position = p_parent + torch.matmul(M_parent, self.joints_offset[joint.name].to(dtype=M_parent.dtype))

        r[..., joint_index_rot, :] = M[..., :2, :].reshape((*M.shape[:-2], 6))
        
        for c in joint.children:
            self._recursive_apply_frame_rot(c, frame_pose, r, M, position)

        return 0
    
    def forward_rot(self, frame_rot):
        #frame_rot: B, F, 54, 3, 3, or BF, 54, 3, 3
        #B = frame_rot.shape[0]
        #p = torch.empty((*frame_rot.shape[:-3], len(self.joints), 3)).to(dtype=frame_rot.dtype, device=frame_rot.device)
        r = torch.empty((*frame_rot.shape[:-2], 6)).to(dtype=frame_rot.dtype, device=frame_rot.device)
        
        M_parent = torch.zeros((*frame_rot.shape[:-3], 3, 3)).to(dtype=frame_rot.dtype, device=frame_rot.device)
        M_parent[..., 0, 0] = 1
        M_parent[..., 1, 1] = 1
        M_parent[..., 2, 2] = 1
        
        self._recursive_apply_frame_rot(self.root, frame_rot, r, M_parent, torch.zeros((*frame_rot.shape[:-3], 3)).to(dtype=frame_rot.dtype, device=frame_rot.device))
        
        return r
    
    def get_global_rot(self, pose_repr):
        
        rot_loc = pose_repr[..., 3:].reshape(*pose_repr.shape[:-1], len(self.joints_wo_end), -1)
        is_euler = rot_loc.shape[-1] == 3
        
        if is_euler:
            # B, F, 54, 3 -> B, F, 54, 3, 3
            rot_loc = euler_angles_to_matrix(rot_loc, 'XYZ')
        else:
            # B, F, 54, 6 -> B, F, 54, 3, 3
            rot_loc = rotation_6d_to_matrix(rot_loc)

        #skeleton = Skeleton(joint_init, dtype=rot_loc.dtype, device=rot_loc.device)
        # rot_loc: B, F, 54, 3, 3 -> B*F, 54, 3, 3
        # out_pose: B*F, 69, 3
        rotation_global = self.forward_rot(rot_loc)
        
        return rotation_global
        
    def forward(self, pose_repr, is_global=True):
        
        # pose_repr: B, F, xxx or B, xxx
        #B = pose_repr.shape[0]
        #F = pose_repr.shape[1]
        
        # B, F, 3 + 54*6
        #if is_root_vel is True:
        #    global_pos = torch.zeros((pose_repr[..., :3].shape)).to(pose_repr.device)
        #    global_pos[..., 1:, :] = torch.cumsum(pose_repr[..., :3], dim=-2)[..., :-1, :]
        #else:
        if is_global is True:
            global_pos = pose_repr[..., :3]
            rot_loc = pose_repr[..., 3:].reshape(*pose_repr.shape[:-1], len(self.joints_wo_end), -1)
        else:
            rot_loc = pose_repr.reshape(*pose_repr.shape[:-1], len(self.joints_wo_end), -1)
                
        is_euler = rot_loc.shape[-1] == 3
        
        if is_euler:
            # B, F, 54, 3 -> B, F, 54, 3, 3
            rot_loc = euler_angles_to_matrix(rot_loc, 'XYZ')
        else:
            # B, F, 54, 6 -> B, F, 54, 3, 3
            rot_loc = rotation_6d_to_matrix(rot_loc)

        #skeleton = Skeleton(joint_init, dtype=rot_loc.dtype, device=rot_loc.device)
        # rot_loc: B, F, 54, 3, 3 -> B*F, 54, 3, 3
        # out_pose: B*F, 69, 3
        out_pose = self.forward_pose(rot_loc)
        #out_pose = out_pose.reshape(B, F, -1, 3)
        
        # mm to m
        # global_pos: B, F, 3
        out_pose = out_pose * 0.001
        
        if is_global is True:
            out_pose += global_pos.unsqueeze(-2)
        
        return out_pose