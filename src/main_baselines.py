import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import torch
import utils.misc as misc
from data.data_utils import MyDataCollate
from torch.utils.data import DataLoader
from utils.dnd_skeleton import *
from data.dnd_config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from data.dnd_dataset import DnDDataset, preprocess_data
from utils.eval_metrics import calculate_eval_metrics 

from utils.dnd_skeleton import Skeleton, BvhJoint

def get_args_parser():
    parser = argparse.ArgumentParser('PolySLGen', add_help=False)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--llama_ckpt_dir", type=str, default='Llama-3-8B-Special-Tokens-Adjusted') #'Llama-3-8B-Special-Tokens-Adjusted')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=float, default=1.0, metavar='N',
                        help='epoch to warmup LR')

    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--max_words', type=int, default=1000)
    
    # ==================== PolySLGen ========================
    parser.add_argument("--data_dir", type=str, default='./train_data/')
    parser.add_argument("--dnd_joint_init_path", type=str, default='')
    
    parser.add_argument('--chunk_length', type=int, default=128)
    parser.add_argument('--hist_length', type=int, default=512)
    parser.add_argument("--pose_hist_length", type=int, default=64)
    
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints/')
    parser.add_argument('--num_body_joints', type=int, default=69)
    parser.add_argument('--val_interval', type=int, default=1) # do validation per xxx epoch
    parser.add_argument('--test_interval', type=int, default=1) # do testing per xxx epoch
    
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--test_ratio', type=float, default=1.0)
                                                          
    parser.add_argument("--steplr_gamma", type=float, default=0.85)
    parser.add_argument("--steplr_step_epoch", type=float, default=1.0)
    
    parser.add_argument("--lora_target_modules", type=str, default=["q_proj", "v_proj", "k_proj"], nargs='+')
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    parser.add_argument("--motion_evaluator_path", type=str, default='./checkpoints/motion_evaluator_exp6_c64_E0150.tar')
    parser.add_argument("--style_dim", type=int, default=256)
    parser.add_argument("--num_style_tokens", type=int, default=1)
    parser.add_argument("--styletts_path", type=str, default='./checkpoints/StyleTTS2/')
    
    parser.add_argument("--pose_fusion", type=int, default=1)
    parser.add_argument("--social_cue", type=int, default=1)
    parser.add_argument("--social_cue_length", type=int, default=2)
    
    parser.add_argument('--loss_mpjpe_weight', type=float, default=1000.0)
    parser.add_argument("--loss_l2_root_weight", type=float, default=50.0)
    parser.add_argument("--loss_audio_weight", type=float, default=100.0)
    parser.add_argument("--loss_pose_weight", type=float, default=0.5)
    parser.add_argument("--loss_reg_weight", type=float, default=10.0)
    parser.add_argument("--loss_turn_weight", type=float, default=0.4)
    
    parser.add_argument("--eval_only", type=int, default=0)
    parser.add_argument("--pretrained_model_dir", type=str, default='')
    
    parser.add_argument("--test_iter", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=16)
    
    return parser

def get_sample(args, chunk_processed_data, index):
    
    input_tokens = torch.tensor(chunk_processed_data['input_tokens'][index]).long()
    raw_dialog = chunk_processed_data['raw_dialog'][index]
    modality_mask = torch.tensor(chunk_processed_data['modality_mask'][index]).long()
    token_role_mask = torch.tensor(chunk_processed_data['token_role_mask'][index]).long()
    
    input_tokens_pad = input_tokens
    target_mask_pad = token_role_mask
    modality_mask_pad = modality_mask
    
    audio_data = {}     
    audio_data_in = chunk_processed_data['audio_data'][index]
    
    for k, v in audio_data_in.items():
        audio_data[k] = torch.from_numpy(v).float()
        
    assert 'gt_data' in audio_data.keys(), 'audio_data is empty'
        
    pose_data = {}
    pose_data_in = chunk_processed_data['pose_data'][index]
    for k, v in pose_data_in.items():
        pose_data[k] = torch.from_numpy(v).float()
        
    assert 'gt_data' in pose_data.keys(), 'pose_data is empty'
   
    labels = copy.deepcopy(input_tokens_pad)
    
    speaking_state = torch.tensor(chunk_processed_data['speaking_state'][index]).float()
    
    return raw_dialog, input_tokens_pad, labels, target_mask_pad, \
        modality_mask_pad, audio_data, pose_data, speaking_state

def get_batch(batch):
    
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
    
    ### audio
    audio_data = {}
    obs_data_all = []
    tgt_data_all = []
    gt_data_all = []
    for b_data in audio_datas:
        obs_data_all.append(b_data['obs_data'])
        tgt_data_all.append(b_data['tgt_data'])
        gt_data_all.append(b_data['gt_data'])
    
    audio_data['obs_data'] = obs_data_all
    audio_data['tgt_data'] = tgt_data_all
    audio_data['gt_data'] = gt_data_all
        
    ### pose
    pose_data = {}
    obs_data_all = []
    tgt_data_all = []
    gt_data_all = []
    for b_data in pose_datas:
        if 'obs_data' in b_data.keys():
            obs_data_all.append(b_data['obs_data'])
        tgt_data_all.append(b_data['tgt_data'])
        gt_data_all.append(b_data['gt_data'])
    
    pose_data['obs_data'] = obs_data_all
    pose_data['tgt_data'] = tgt_data_all
    pose_data['gt_data'] = gt_data_all
        
    return raw_dialogs, input_tokens_pads, labelss, target_mask_pads, modality_mask_pads, audio_data, pose_data, speaking_states
            
def get_gts(input_tokens, target_mask, modality_mask, audio_data, pose_data, body_skeleton, tokenizer):
    
    ###########
    # mask the labels with target_mask_pad, only caclulate loss for the target subject (non-history)
    bsz = len(input_tokens)
    
    # find the total_len and left_pad and right_pad
    # check left
    start_index = [torch.where(t==1)[0][0] for t in target_mask]
    min_prompt_len = max([t for i, t in zip(input_tokens, start_index)])
    max_prompt_len_right = max([len(i)-t for i, t in zip(input_tokens, start_index)])
    total_len = min_prompt_len + max_prompt_len_right
    
    input_tokens_pad = torch.full((bsz, total_len), -1).cuda().long()
    target_mask_pad = torch.full((bsz, total_len), 0).cuda().long()
    modality_mask_pad = torch.full((bsz, total_len), -1).cuda().long()
    
    # all input_tokens should NOT have the same length
    for k, (t, s) in enumerate(zip(input_tokens, start_index)):
        input_tokens_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
    
    for k, (t, s) in enumerate(zip(target_mask, start_index)):
        target_mask_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
        
    for k, (t, s) in enumerate(zip(modality_mask, start_index)):
        modality_mask_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
        
    # take all gt_data, will filter out the dummmy ones when saveing the final gt
    target_gt_audio_data = audio_data['gt_data']
    target_gt_pose_data = pose_data['gt_data']
        
    #### text ####
    gt_text_tokens_out = []
    for input_tokens_pad_, target_mask_pad_ in zip(input_tokens_pad, target_mask_pad):
        tgt_ids = torch.where((target_mask_pad_ == 1) & (input_tokens_pad_ != tokenizer.pose_token_id) & (input_tokens_pad_ != tokenizer.scue_token_id) & (input_tokens_pad_ != tokenizer.audio_token_id))
        if len(tgt_ids[0]) != 0:
            gt_text_tokens_out.append(input_tokens_pad_[tgt_ids].detach().float().cpu().numpy())
        else:
            gt_text_tokens_out.append([])
    
    #### audio ####   
    gt_audio_tokens_out = []
    for target_gt_audio_data_, target_mask_pad_, input_tokens_pad_ in zip(target_gt_audio_data, target_mask_pad, input_tokens_pad):
        gt_audio_ids = torch.where((target_mask_pad_ == 1) & (input_tokens_pad_ == tokenizer.audio_token_id))
        if len(gt_audio_ids[0]) != 0:
            gt_audio_tokens_out.append(target_gt_audio_data_.detach().float().cpu().numpy())
        else: 
            gt_audio_tokens_out.append([])
        
    #### pose, needs to reconstruct wholebody pose ####
    gt_wholebody_pose_out = []
    gt_repr_pose_out = []
    if len(target_gt_pose_data) != 0:
        # pred_pose: B, TP-target, 69, 3
        # unnorm
        for gt_pose_unnorm_ in target_gt_pose_data:
            gt_wholebody_pose_ = body_skeleton.forward(gt_pose_unnorm_)
            gt_wholebody_pose_out += [gt_wholebody_pose_.detach().float().cpu().numpy()]
            gt_repr_pose_out += [gt_pose_unnorm_.detach().float().cpu().numpy()]
            
    gts = {'text_tok': gt_text_tokens_out,
           'audio_emb': gt_audio_tokens_out, 
           'pose': gt_wholebody_pose_out,
           'pose_repr': gt_repr_pose_out}
    
    return gts
    
def evaluation_random(model, tokenizer, data_loader_test, chunk_train_data, args=None, mode='val', test_logger=None):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '***{}***'.format(mode)
    print_freq = 1
    
    rank = next(model.parameters()).device
    
    joint_init = np.load(f'{args.dnd_joint_init_path}', allow_pickle=True).flat[0]
    body_skeleton = Skeleton(joint_init)
    body_skeleton.cuda()
        
    start_iter = 0
    final_outputs = []
    with torch.inference_mode():
        for _, data  in enumerate(
            metric_logger.log_every(data_loader_test, print_freq, header, start_iter, logger=test_logger), start=start_iter
        ):
            raw_dialogue, input_tokens, labels, target_mask, modality_mask, audio_data, pose_data, speaking_state = data
            
            # List of tensor: input_tokens_pad, labels, target_mask_pad
            input_tokens = [ele.to(device=rank).long() for ele in input_tokens]
            labels = [ele.to(device=rank).long() for ele in labels]
            target_mask = [ele.to(device=rank).long() for ele in target_mask]
            modality_mask = [ele.to(device=rank).long() for ele in modality_mask]
            speaking_state = [ele.to(device=rank).float() for ele in speaking_state]
            
            for k, v in audio_data.items():
                if isinstance(audio_data[k], list):
                    audio_data[k] = [ ele.to(device=rank) for ele in audio_data[k]]
                else:
                    audio_data[k] = audio_data[k].to(device=rank)
                
            for k, v in pose_data.items():
                if isinstance(pose_data[k], list):
                    pose_data[k] = [ ele.to(device=rank) for ele in pose_data[k]]
                else:
                    pose_data[k] = pose_data[k].to(device=rank)
            
            batch_size = len(input_tokens)
            gts = get_gts(input_tokens, target_mask, modality_mask, audio_data, pose_data, body_skeleton, tokenizer)
            
            # -------
            #  Prediction
            # -------
            # ranomdly sample from traiing data
            sampled_batch = []
            index = torch.randint(low=0, high=len(chunk_train_data), size=(batch_size,))
            for index_ in index:
                sampled_batch += [get_sample(args, chunk_train_data, index_)]
            
            _, sampled_input_tokens, sampled_labels, sampled_target_mask, sampled_modality_mask, sampled_audio_data, sampled_pose_data, sampled_speaking_state = get_batch(sampled_batch)
            
            # List of tensor: input_tokens_pad, labels, target_mask_pad
            sampled_input_tokens = [ele.to(device=rank).long() for ele in sampled_input_tokens]
            sampled_labels = [ele.to(device=rank).long() for ele in sampled_labels]
            sampled_target_mask = [ele.to(device=rank).long() for ele in sampled_target_mask]
            sampled_modality_mask = [ele.to(device=rank).long() for ele in sampled_modality_mask]
            sampled_speaking_state = [ele.to(device=rank).float() for ele in sampled_speaking_state]
            
            for k, v in sampled_audio_data.items():
                if isinstance(sampled_audio_data[k], list):
                    sampled_audio_data[k] = [ ele.to(device=rank) for ele in sampled_audio_data[k]]
                else:
                    sampled_audio_data[k] = sampled_audio_data[k].to(device=rank)
                
            for k, v in sampled_pose_data.items():
                if isinstance(pose_data[k], list):
                    sampled_pose_data[k] = [ ele.to(device=rank) for ele in sampled_pose_data[k]]
                else:
                    sampled_pose_data[k] = sampled_pose_data[k].to(device=rank)
                    
            preds = get_gts(sampled_input_tokens, sampled_target_mask, sampled_modality_mask, sampled_audio_data, sampled_pose_data, body_skeleton, tokenizer)
            
            # save testing results
            final_outputs.append({'preds': preds, 'gts': gts, 'raw_dialogue': raw_dialogue})
    
    test_logger.info(f"Averaged stats: {metric_logger}")
    return final_outputs


def get_modalities(input_tokens, target_mask, audio_data, pose_data, body_skeleton, tokenizer):
    text = []
    for input_tokens_, target_mask_, in zip(input_tokens, target_mask):
        tgt_text_ids = torch.where((input_tokens_ != tokenizer.pose_token_id) & (input_tokens_ != tokenizer.scue_token_id) & (input_tokens_ != tokenizer.audio_token_id) & (target_mask_ != 1))
        text += [input_tokens_[tgt_text_ids]]
     
    style = []   
    for audio_data_ in audio_data['obs_data']:
        style += [audio_data_]
    
    pose = []   
    for pose_data_ in pose_data['obs_data']:
        pose += [body_skeleton(pose_data_)]
            
    return text, style, pose
   
from evaluator.utils.motion_process import process_motion, decompose 
def get_embeddings(texts, styles, poses, text_embedder, movement_enc, mean, std): 
     
    def to_motion(pred_, mean, std, feat_dim):
        
        data_unnorm = torch.from_numpy(process_motion(pred_.cpu().numpy()))
        g_rot, motion = decompose(data_unnorm)
        motion = ((motion - mean) / std).reshape(-1, feat_dim-6)
        pred_motion = torch.cat((g_rot, motion), dim=-1) # T' K
        
        return pred_motion
    
    dim_pose = 207 + 6
    text_embs = []
    pose_embs = []
    for texts_, poses_ in zip(texts, poses):
        
        # L1, 3072
        text_emb = text_embedder(texts_)
        text_embs += [text_emb]
        
        # poses_: 32*5, 327
        poses_all = poses_.reshape(5, -1, poses_.shape[-2], poses_.shape[-1])
        
        pose_emb = []
        for pose_ in poses_all:
            pred_motion = to_motion(pose_, mean, std, dim_pose) # T' K
            pose_emb += [torch.cat((pred_motion, pred_motion[-1:]), dim=0)]
            
        pose_emb = torch.stack(pose_emb, dim=0)  
        
        # 5*L2, 512   
        pose_emb = movement_enc(pose_emb.cuda())
        pose_emb = pose_emb.reshape(-1, pose_emb.shape[-1])
        pose_embs += [pose_emb]
    
    return text_embs, pose_embs
 
from torch.nn import CosineSimilarity   
def find_nearest_samples(curr_condition_emb, train_condition_embs):
    
    text_embs, pose_embs = curr_condition_emb
    train_text_embs, train_pose_embs = train_condition_embs
    
    similarity = CosineSimilarity(dim=-1, eps=1e-6)
    indexs = []
    # text_emb: L1, 3072
    # pose_emb: L3, 512
    for text_emb, pose_emb in zip(text_embs, pose_embs):
        scores = []
        # text_emb: L1', 3072
        # pose_emb: L3, 512
        for train_text_emb, train_pose_emb in zip(train_text_embs, train_pose_embs):
            total_len = max(text_emb.shape[0], train_text_emb.shape[0])
            text_emb_pad = torch.full((total_len, text_emb.shape[-1]), 0).cuda().float()
            text_emb_pad[:len(text_emb)] = text_emb
            train_text_emb_pad = torch.full((total_len, train_text_emb.shape[-1]), 0).cuda().float()
            train_text_emb_pad[:len(train_text_emb)] = train_text_emb
            
            text_sim = similarity(text_emb_pad, train_text_emb_pad).mean()
            pose_sim = similarity(pose_emb, train_pose_emb).mean()
            
            scores += [pose_sim+text_sim]
        
        scores = torch.stack(scores, dim=0)
        indexs += [torch.argmax(scores, dim=0)]
    
    return indexs

def evaluation_nn(model, tokenizer, data_loader_test, chunk_train_data, movement_enc, args=None, mode='val', test_logger=None, mean = None, std = None):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '***{}***'.format(mode)
    print_freq = 1
    
    rank = next(model.parameters()).device
    
    joint_init = np.load(f'{args.dnd_joint_init_path}', allow_pickle=True).flat[0]
    body_skeleton = Skeleton(joint_init)
    body_skeleton.cuda()
    
    texts = []
    styles = []
    poses = []
    for i in range(len(chunk_train_data)):
        sampled_batch = [get_sample(args, chunk_train_data, i)]
        _, input_tokens, labels, target_mask, modality_mask, audio_data, pose_data, speaking_state = get_batch(sampled_batch)
        
        input_tokens = [ele.to(device=rank).long() for ele in input_tokens]
        labels = [ele.to(device=rank).long() for ele in labels]
        target_mask = [ele.to(device=rank).long() for ele in target_mask]
        modality_mask = [ele.to(device=rank).long() for ele in modality_mask]
        speaking_state = [ele.to(device=rank).float() for ele in speaking_state]
        
        for k, v in audio_data.items():
            if isinstance(audio_data[k], list):
                audio_data[k] = [ ele.to(device=rank) for ele in audio_data[k]]
            else:
                audio_data[k] = audio_data[k].to(device=rank)
            
        for k, v in pose_data.items():
            if isinstance(pose_data[k], list):
                pose_data[k] = [ ele.to(device=rank) for ele in pose_data[k]]
            else:
                pose_data[k] = pose_data[k].to(device=rank)
        
        text, style, pose = get_modalities(input_tokens, target_mask, audio_data, pose_data, body_skeleton, tokenizer)
        
        texts += text
        styles += style
        poses += pose
      
    train_condition_embs = get_embeddings(texts, styles, poses, model.get_input_embeddings(), movement_enc, mean, std)
        
    start_iter = 0
    final_outputs = []
    with torch.inference_mode():
        for _, data  in enumerate(
            metric_logger.log_every(data_loader_test, print_freq, header, start_iter, logger=test_logger), start=start_iter
        ):
            raw_dialogue, input_tokens, labels, target_mask, modality_mask, audio_data, pose_data, speaking_state = data
            
            # List of tensor: input_tokens_pad, labels, target_mask_pad
            input_tokens = [ele.to(device=rank).long() for ele in input_tokens]
            labels = [ele.to(device=rank).long() for ele in labels]
            target_mask = [ele.to(device=rank).long() for ele in target_mask]
            modality_mask = [ele.to(device=rank).long() for ele in modality_mask]
            speaking_state = [ele.to(device=rank).float() for ele in speaking_state]
            
            for k, v in audio_data.items():
                if isinstance(audio_data[k], list):
                    audio_data[k] = [ ele.to(device=rank) for ele in audio_data[k]]
                else:
                    audio_data[k] = audio_data[k].to(device=rank)
                
            for k, v in pose_data.items():
                if isinstance(pose_data[k], list):
                    pose_data[k] = [ ele.to(device=rank) for ele in pose_data[k]]
                else:
                    pose_data[k] = pose_data[k].to(device=rank)
            
            gts = get_gts(input_tokens, target_mask, modality_mask, audio_data, pose_data, body_skeleton, tokenizer)
            
            text, style, pose = get_modalities(input_tokens, target_mask, audio_data, pose_data, body_skeleton, tokenizer)
            curr_condition_emb = get_embeddings(text, style, pose, model.get_input_embeddings(), movement_enc, mean, std)
            
            # -------
            #  Prediction
            # -------
            # ranomdly sample from traiing data
            sampled_batch = []
            # select samples that are closest to the condition
            index = find_nearest_samples(curr_condition_emb, train_condition_embs)
            for index_ in index:
                sampled_batch += [get_sample(args, chunk_train_data, index_)]
            
            _, sampled_input_tokens, sampled_labels, sampled_target_mask, sampled_modality_mask, sampled_audio_data, sampled_pose_data, sampled_speaking_state = get_batch(sampled_batch)
            
            # List of tensor: input_tokens_pad, labels, target_mask_pad
            sampled_input_tokens = [ele.to(device=rank).long() for ele in sampled_input_tokens]
            sampled_labels = [ele.to(device=rank).long() for ele in sampled_labels]
            sampled_target_mask = [ele.to(device=rank).long() for ele in sampled_target_mask]
            sampled_modality_mask = [ele.to(device=rank).long() for ele in sampled_modality_mask]
            
            sampled_speaking_state = [ele.to(device=rank).float() for ele in sampled_speaking_state]
            
            for k, v in sampled_audio_data.items():
                if isinstance(sampled_audio_data[k], list):
                    sampled_audio_data[k] = [ ele.to(device=rank) for ele in sampled_audio_data[k]]
                else:
                    sampled_audio_data[k] = sampled_audio_data[k].to(device=rank)
                
            for k, v in sampled_pose_data.items():
                if isinstance(pose_data[k], list):
                    sampled_pose_data[k] = [ ele.to(device=rank) for ele in sampled_pose_data[k]]
                else:
                    sampled_pose_data[k] = sampled_pose_data[k].to(device=rank)
                    
            preds = get_gts(sampled_input_tokens, sampled_target_mask, sampled_modality_mask, sampled_audio_data, sampled_pose_data, body_skeleton, tokenizer)
            
            
            # save testing results
            final_outputs.append({'preds': preds, 'gts': gts, 'raw_dialogue': raw_dialogue})
    
    test_logger.info(f"Averaged stats: {metric_logger}")
    return final_outputs
    
from evaluator.network import MovementConvEncoder
def eval_main_retrieve(args, tokenizer):
    
    test_logger = misc.get_logger(args.output_dir, 'test_log')
     
    args.gpu = 0
    args.distributed = False
    torch.cuda.set_device(0)
    
    test_logger.info("Loading pretrained weights ...")
    
    # only use the embedding layer for text embeddings
    model = AutoModelForCausalLM.from_pretrained(f'{args.checkpoint_dir}/{args.llama_ckpt_dir}',
                                                    quantization_config=None,
                                                    device_map="cpu",
                                                    attn_implementation="eager", #"sdpa", 
                                                    torch_dtype=torch.bfloat16,
                                                    use_cache=True, 
                                                    pad_token_id = tokenizer.pad_token_id)
    for name, param in model.get_input_embeddings().named_parameters():
        param.data[tokenizer.pad_token_id] = torch.zeros_like(param[:1])
       
    model.to(device='cuda')
    model.eval()
    test_logger.info(f"Model = {str(model)}")
    
    # load motion embedder
    checkpoint = torch.load(args.motion_evaluator_path, map_location="cpu", weights_only=False)
    dim_pose = 207 + 6
    dim_movement_enc_hidden = 512
    
    stat_dir = os.path.join(os.path.dirname(args.motion_evaluator_path), '../')
    mean = torch.from_numpy(np.load(f'{stat_dir}/mean.npy', allow_pickle=True))
    std = torch.from_numpy(np.load(f'{stat_dir}/std.npy', allow_pickle=True))
          
    movement_enc = MovementConvEncoder(dim_pose, dim_movement_enc_hidden, dim_movement_enc_hidden)
    movement_enc.load_state_dict(checkpoint['movement_enc'])
    movement_enc.cuda()
    movement_enc.eval()
    
    set_name = 'test'
    chunk_train_data = preprocess_data(args, tokenizer, partition='train', logger=test_logger, is_baseline = True)
    chunk_test_data = preprocess_data(args, tokenizer, partition='test', logger=test_logger, is_baseline = True)
    mydatacollate = MyDataCollate(args=args)
    dataset_test = DnDDataset(args, chunk_test_data, tokenizer, partition = 'test')
    test_logger.info(f'[Test] dataset_{set_name} size: {len(dataset_test)}')
    
    data_loader_test = DataLoader(
        dataset_test,
        batch_size= args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
        collate_fn=mydatacollate.test_collate_fn,
    )
    test_logger.info(f'[Test] data_loader_{set_name} size: {len(data_loader_test)}')
    
    set_name = 'test'   
       
    for cur_mode in ['random', 'nn'] :
        
        output_dir_mode = os.path.join(args.output_dir, cur_mode)
        os.makedirs(output_dir_mode, exist_ok=True)

        for ii in range(args.test_iter):
            
            if os.path.exists(os.path.join(output_dir_mode, f"{set_name}_results_{ii}.npy")) is True:
                continue
            
            # evaluation
            start_test_time = time.time()
            if cur_mode == 'random':
                final_outputs = evaluation_random(
                    model,
                    tokenizer,
                    data_loader_test,
                    chunk_train_data,
                    args=args,
                    mode='test',
                    test_logger=test_logger
                    )
            else:
                final_outputs = evaluation_nn(
                    model,
                    tokenizer,
                    data_loader_test,
                    chunk_train_data,
                    movement_enc, 
                    args=args,
                    mode='test',
                    test_logger=test_logger,
                    mean=mean, 
                    std=std
                    )
            
            total_test_time = time.time() - start_test_time
            total_test_time_str = str(datetime.timedelta(seconds=int(total_test_time)))
            test_logger.info('Testing time {}'.format(total_test_time_str))

            test_log_stats = {'time': total_test_time_str}
            
            if output_dir_mode and misc.is_main_process():
                with open(os.path.join(output_dir_mode, f"{set_name}_log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(test_log_stats) + "\n") 
                torch.save(final_outputs, os.path.join(output_dir_mode, f"{set_name}_results_{ii}.npy"))
            
    for handler in test_logger.handlers:
        test_logger.removeHandler(handler)
        handler.close()

def main(args):
    
    with open(f'{args.output_dir}/run_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    tokenizer = AutoTokenizer.from_pretrained(f'{args.checkpoint_dir}/{args.llama_ckpt_dir}')
    
    # ---- update tokenizer -------
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.padding_side = "left"
    
    pose_token = "<|reserved_special_token_243|>"
    audio_token = "<|reserved_special_token_242|>"
    scue_token = "<|reserved_special_token_236|>"
    pad_token = "<|reserved_special_token_250|>"
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [pose_token, audio_token, scue_token, pad_token]
    })
    
    tokenizer.pose_token = pose_token
    tokenizer.pose_token_id = tokenizer.convert_tokens_to_ids(pose_token)
    
    tokenizer.audio_token = audio_token
    tokenizer.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    
    tokenizer.scue_token = scue_token
    tokenizer.scue_token_id = tokenizer.convert_tokens_to_ids(scue_token)
    
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    # -------------------------------
    
    eval_main_retrieve(args, tokenizer)
    
    # calculate metrics for each iteration
    calculate_eval_metrics(args, tokenizer, state_from_text=True, baseline_mode='random')
    calculate_eval_metrics(args, tokenizer, state_from_text=True, baseline_mode='nn')
  
       
if __name__ == '__main__':
    
    misc.set_random_seeds(42)
    
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)