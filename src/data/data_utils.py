from data.dnd_config import *
import torch

class MyDataCollate():
    def __init__(self, args):
        self.args =args
    
    def train_collate_fn(self, batch):     
        
        raw_dialogs = list()
        input_tokens_pads = list()
        labelss = list()
        target_mask_pads = list()
        modality_mask_pads = list()
        audio_datas = list()
        pose_datas = list()
        speaking_states = list()

        for b in batch:
            raw_dialogs.append(b[0])
            input_tokens_pads.append(b[1])
            labelss.append(b[2])
            target_mask_pads.append(b[3])
            modality_mask_pads.append(b[4])
            audio_datas.append(b[5])
            pose_datas.append(b[6])
            speaking_states.append(b[7])
        
        raw_dialog = raw_dialogs
            
        input_tokens_pad = torch.stack(input_tokens_pads, dim=0)
        labels = torch.stack(labelss, dim=0)
        target_mask_pad = torch.stack(target_mask_pads, dim=0)
        modality_mask_pad = torch.stack(modality_mask_pads, dim=0)
        speaking_state = torch.stack(speaking_states, dim=0)
        
        ### enable audio
        audio_data = {}
        obs_data_all = []
        tgt_data_all = []
        gt_data_all = []
        for b_data in audio_datas:
            if 'obs_data' in b_data.keys():
                obs_data_all.append(b_data['obs_data'])
            tgt_data_all.append(b_data['tgt_data'])
            gt_data_all.append(b_data['gt_data'])
        
        audio_data['obs_data'] = obs_data_all
        audio_data['tgt_data'] = tgt_data_all
        audio_data['gt_data'] = gt_data_all
            
        ### Enable pose
        pose_data = {}
        obs_data_all = []
        interact_data_all = []
        tgt_data_all = []
        gt_data_all = []
        for b_data in pose_datas:
            if 'obs_data' in b_data.keys():
                obs_data_all.append(b_data['obs_data'])
            if 'interact_data' in b_data.keys():
                interact_data_all.append(b_data['interact_data'])
            tgt_data_all.append(b_data['tgt_data'])
            gt_data_all.append(b_data['gt_data'])
        
        pose_data['obs_data'] = obs_data_all
        pose_data['interact_data'] = interact_data_all
        pose_data['tgt_data'] = tgt_data_all
        pose_data['gt_data'] = gt_data_all
        
        return raw_dialog, input_tokens_pad, labels, target_mask_pad, modality_mask_pad, audio_data, pose_data, speaking_state
    
    def test_collate_fn(self, batch):     
        
        raw_dialogs = list()
        input_tokens_pads = list()
        labelss = list()
        target_mask_pads = list()
        modality_mask_pads = list()
        audio_datas = list()
        pose_datas = list()
        speaking_states = list()

        for b in batch:
            raw_dialogs.append(b[0])
            input_tokens_pads.append(b[1])
            labelss.append(b[2])
            target_mask_pads.append(b[3])
            modality_mask_pads.append(b[4])
            audio_datas.append(b[5])
            pose_datas.append(b[6])
            speaking_states.append(b[7])
        
        ### enable audio
        audio_data = {}
        obs_data_all = []
        tgt_data_all = []
        gt_data_all = []
        for b_data in audio_datas:
            if 'obs_data' in b_data.keys():
                obs_data_all.append(b_data['obs_data'])
            tgt_data_all.append(b_data['tgt_data'])
            gt_data_all.append(b_data['gt_data'])
        
        audio_data['obs_data'] = obs_data_all
        audio_data['tgt_data'] = tgt_data_all
        audio_data['gt_data'] = gt_data_all
            
        ### Enable pose
        pose_data = {}
        obs_data_all = []
        interact_data_all = []
        tgt_data_all = []
        gt_data_all = []
        for b_data in pose_datas:
            if 'obs_data' in b_data.keys():
                obs_data_all.append(b_data['obs_data'])
            if 'interact_data' in b_data.keys():
                interact_data_all.append(b_data['interact_data'])
            tgt_data_all.append(b_data['tgt_data'])
            gt_data_all.append(b_data['gt_data'])
        
        pose_data['obs_data'] = obs_data_all
        pose_data['interact_data'] = interact_data_all
        pose_data['tgt_data'] = tgt_data_all
        pose_data['gt_data'] = gt_data_all
        
            
        return raw_dialogs, input_tokens_pads, labelss, target_mask_pads, modality_mask_pads, audio_data, pose_data, speaking_states
