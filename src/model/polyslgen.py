import torch
import torch.nn as nn

import numpy as np
from typing import Optional, Tuple

from utils.recon_helper import *
from utils.dnd_skeleton import Skeleton, BvhJoint
import copy
from einops import rearrange

from transformers import DynamicCache
from utils.eval_similarity import *

# We use a wrapper to keep llama model untouched
class PolySLGen(nn.Module):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # ========================
        # Additional modules/layers we need for more modalities
        # ========================
        llm_hidden_size = model.config.hidden_size
        self.llm_hidden_size = llm_hidden_size
        resample_dim =1024
        self.pose_dim = 327
        
        self.input_emb = nn.ModuleDict()
        self.proj1 = nn.ModuleDict()
        self.proj2 = nn.ModuleDict()
        self.decode = nn.ModuleDict()
        
        self.modals = ['audio', 'body']
        
        if args.pose_fusion != 0:
            self.modals += ['body_fusion']
            
        if args.social_cue != 0:
            self.modals += ['social_cue']
                
        for modal in self.modals:
            
            # ========================
            # Adapters for each modality
            # ========================
            # body: B, T=64, 3, J=69 --> B, 3, 69, T=32 --> 
            # audio: B, T', 8 --> B, 1, 8, T' --> 
            
            if modal == 'body':
                # B, ..., 327 --> B, ..., 1024 --> B, ..., llm_hidden_size
                self.input_emb[modal] =  nn.Linear(self.pose_dim, resample_dim)
                self.proj1[modal] = nn.Sequential(
                            nn.Linear(resample_dim, resample_dim),
                            nn.LayerNorm(resample_dim)
                            )
                self.proj2[modal] = nn.Sequential(
                            nn.Linear(resample_dim, llm_hidden_size),
                            nn.LayerNorm(llm_hidden_size)
                            )
            
            elif modal == 'body_fusion':
                
                from utils.attention import SpatialTransformer, ResBlock
                
                # B, ..., 327 --> B, ..., 512 
                self.input_emb[modal] = nn.Sequential(
                                nn.Conv1d(self.pose_dim, resample_dim//2, kernel_size=1, stride=1, padding=0),
                                ResBlock(channels=resample_dim//2, 
                                                emb_channels=resample_dim//2,
                                                dropout=0.0,
                                                out_channels=resample_dim//2,
                                            dims=1,
                                                ))
                
                # B, ..., 327  --> B, ..., 512
                self.context_emb = nn.Sequential(
                                nn.Linear(self.pose_dim, resample_dim//2),
                                nn.GELU(),
                                nn.Linear(resample_dim//2, resample_dim//2)
                            )
                
                n_heads = 8
                dim_head = (resample_dim//2) // n_heads
                self.proj1[modal] = SpatialTransformer(in_channels=resample_dim//2, 
                                                            n_heads = 8, 
                                                            d_head = dim_head, 
                                                            depth = 1, 
                                                            context_dim = resample_dim//2)
                
                # B, ..., 512 --> B, ..., llm_hidden_size
                self.proj2[modal] = nn.Sequential(
                            nn.Linear(resample_dim//2, llm_hidden_size),
                            nn.LayerNorm(llm_hidden_size),
                            nn.Linear(llm_hidden_size, llm_hidden_size))
            
            elif modal == 'social_cue':
            
                dim_emb = 16
                
                # B, ..., 1 --> B, ..., 16
                self.interact_emb = nn.Linear(1, dim_emb)
                  
                # B, 64, ... --> B, 2, ...
                self.input_emb[modal] = nn.Sequential(
                            nn.Conv2d(args.pose_hist_length, args.social_cue_length, kernel_size=3, padding=1),
                            nn.GroupNorm(1, args.social_cue_length),
                            )
                
                # B, ..., 4*32 --> B, ..., 1024
                self.proj1[modal] = nn.Sequential(
                            nn.Linear(4 * dim_emb, resample_dim),
                            nn.LayerNorm(resample_dim),
                            nn.Linear(resample_dim, resample_dim),
                            )
                
                # B, ..., 1024 --> B, ..., llm_hidden_size
                self.proj2[modal] = nn.Sequential(
                        nn.Linear(resample_dim, llm_hidden_size),
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, llm_hidden_size),
                        )
                                      
            elif modal == 'audio':
                
                # B, ..., 256 --> B, ..., 1024
                self.input_emb[modal] = nn.Sequential(
                        nn.Linear(args.style_dim, resample_dim),
                        nn.LayerNorm(resample_dim))
                   
                # B, ..., 1024 --> B, ..., 1024
                self.proj1[modal] = nn.Sequential(
                    nn.Linear(resample_dim, resample_dim),
                    nn.LayerNorm(resample_dim))
                    
                # B, ..., 1024 --> B, ..., llm_hidden_size
                self.proj2[modal] = nn.Sequential(
                            nn.Linear(resample_dim, llm_hidden_size),
                            nn.LayerNorm(llm_hidden_size))
            
            
            #### predict speaking-state score
            # B, 3072 --> B, 1
            self.pred_state = nn.Sequential(
                        nn.Linear(llm_hidden_size, resample_dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(resample_dim//2, 32),
                        nn.ReLU(inplace=True),
                        nn.Linear(32, 1),)
            
            # ========================
            # decode output embeddings to target modality
            # ========================
            if modal == 'body':
                self.decode[modal] = nn.Sequential(nn.Linear(llm_hidden_size, llm_hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(llm_hidden_size, self.pose_dim),
                                                    nn.Dropout(0.0),)
            elif modal == 'audio':
                self.decode[modal] = nn.Sequential(nn.Linear(llm_hidden_size, llm_hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(llm_hidden_size, args.style_dim),
                                            nn.Dropout(0.0),)
            elif modal == 'state':
                self.decode[modal] = nn.Sequential(nn.Linear(llm_hidden_size, llm_hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(llm_hidden_size, 1),
                                                    nn.Dropout(0.0),)
            
        # this is the LLM backbone
        self.llama = model
     
    def encode_modality(self, x, modal='body'):
        
        if modal in ['body']:
            x = self.input_emb[modal](x)
            x = self.proj1[modal](x)
            out = self.proj2[modal](x)
         
        elif modal in ['body_fusion']:
            
            #B = x.shape[0]
            D = x.shape[-1]
            T = self.args.pose_hist_length
            x_in = x[:, -T:].reshape(-1, D, 1)
            context = x[:, :-T].reshape(-1, 4, T, D).permute(0, 2, 1, 3).reshape(-1, 4, D)
            
            x_in = self.input_emb[modal](x_in)
            context = self.context_emb(context)
            
            x_in = self.proj1[modal](x_in, context=context)
            out = self.proj2[modal](x_in.squeeze(-1)).reshape(-1, T, self.llm_hidden_size)
            
        elif modal in ['social_cue']:
            
            x = self.interact_emb(x)
            x = self.input_emb[modal](x)
            x = self.proj1[modal](x.flatten(start_dim=-2))
            out = self.proj2[modal](x)
                            
        elif modal in ['audio']:
            
            B = x.shape[0]
            x = self.input_emb[modal](x).reshape(B, -1)
            x = self.proj1[modal](x)
            out = self.proj2[modal](x)
                
        return out
    
    def decode_modality(self, x, modal='body'):
        
        x = self.decode[modal](x)
            
        return x
    
    def forward(self, input_tokens_pad, target_mask_pad, modality_mask_pad, audio_data, pose_data, labels=None):
        
        pred_speaking_state = []
        
        # -------------
        # Text embeddings
        # -------------
        attention_mask = torch.where(input_tokens_pad==self.pad_token_id, 0, 1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        input_embeddings = self.llama.get_input_embeddings()(input_tokens_pad)
        
        # -------------
        # Encode/insert audio embeddings
        # -------------
        target_gt_audio_data = []
        if len(audio_data['gt_data']) != 0:
            
            obs_batch_audio_data = []
            if len(audio_data['obs_data']) != 0:
                obs_batch_audio_data = torch.cat(audio_data['obs_data'], dim=0)
                
                obs_audio_ids = torch.where((labels == self.tokenizer.audio_token_id) & (target_mask_pad != 1))
                obs_audio_embeddings = self.encode_modality(obs_batch_audio_data, modal='audio')
                input_embeddings[obs_audio_ids] = obs_audio_embeddings.to(dtype=input_embeddings.dtype)
                
            tgt_batch_audio_data = []
            if len(audio_data['tgt_data']) != 0:
                
                for labels_, target_mask_pad_, tgt_data_, gt_data_ in zip(labels, target_mask_pad, audio_data['tgt_data'], audio_data['gt_data']):
                    tgt_audio_ids_ = torch.where((labels_ == self.tokenizer.audio_token_id) & (target_mask_pad_ == 1))
                    if len(tgt_audio_ids_[0]) != 0:
                        tgt_batch_audio_data += [tgt_data_]
                        target_gt_audio_data += [gt_data_]
                 
                if len(tgt_batch_audio_data) != 0:
                    tgt_batch_audio_data = torch.cat(tgt_batch_audio_data, dim=0)
                    target_gt_audio_data = torch.cat(target_gt_audio_data, dim=0)
                
                tgt_audio_ids = torch.where((labels == self.tokenizer.audio_token_id) & (target_mask_pad == 1))
                if len(tgt_audio_ids[0]) != 0:
                    tgt_audio_embeddings = self.encode_modality(tgt_batch_audio_data, modal='audio')
                    input_embeddings[tgt_audio_ids] = tgt_audio_embeddings.to(dtype=input_embeddings.dtype)
               
        # -------------
        # Encode/insert pose embeddings
        # -------------
        target_gt_pose_data = []
        if len(pose_data['gt_data']) != 0:
            
            # ---- gt pose ----
            target_gt_pose_data = torch.cat(pose_data['gt_data'], dim=0)
            
            # ---- observed pose embeddings ---- 
            if self.args.pose_fusion != 0:
                
                obs_batch_pose_data = []
                if len(pose_data['obs_data']) != 0:
                    
                    obs_batch_pose_data = torch.cat(pose_data['obs_data'], dim=0)
                    obs_batch_pose_data = rearrange(obs_batch_pose_data, '(b t) d -> b t d', t = 5 * self.args.pose_hist_length)
                    
                    # obs_batch_pose_data: B, 5T, 327  -- joint leanring --> B, T, 327 --> BT, 327
                    obs_pose_ids = torch.where((labels == self.tokenizer.pose_token_id) & (target_mask_pad != 1))
                    obs_pose_embeddings = self.encode_modality(obs_batch_pose_data, modal='body_fusion')
                    obs_pose_embeddings = rearrange(obs_pose_embeddings, 'b t d -> (b t) d')
                    input_embeddings[obs_pose_ids] = obs_pose_embeddings.to(dtype=input_embeddings.dtype)
                    
            else:
                obs_batch_pose_data = []
                if len(pose_data['obs_data']) != 0:
                    
                    obs_batch_pose_data = torch.cat(pose_data['obs_data'], dim=0)
                    obs_pose_ids = torch.where((labels == self.tokenizer.pose_token_id) & (target_mask_pad != 1))
                    obs_pose_embeddings = self.encode_modality(obs_batch_pose_data, modal='body')
                    input_embeddings[obs_pose_ids] = obs_pose_embeddings.to(dtype=input_embeddings.dtype)
            
            # ---- social cue embeddings ----         
            if self.args.social_cue != 0:
                interact_feat = []
                if len(pose_data['interact_data']) != 0:
                    interact_feat = torch.stack(pose_data['interact_data'], dim=0)
            
                interact_embeddings = self.encode_modality(interact_feat, modal='social_cue')
                interact_ids = torch.where((labels == self.tokenizer.scue_token_id) & (target_mask_pad != 1))
                interact_embeddings = rearrange(interact_embeddings, 'b t d -> (b t) d')
                input_embeddings[interact_ids] = interact_embeddings.to(dtype=input_embeddings.dtype)    
                            
            # ---- target pose embeddings ---- 
            tgt_batch_pose_data = []
            if len(pose_data['tgt_data']) != 0:
                
                tgt_batch_pose_data = torch.cat(pose_data['tgt_data'], dim=0)
                tgt_pose_ids = torch.where((labels == self.tokenizer.pose_token_id) & (target_mask_pad == 1))
                tgt_pose_embeddings = self.encode_modality(tgt_batch_pose_data, modal='body')#.reshape(B, TP, -1)
                input_embeddings[tgt_pose_ids] = tgt_pose_embeddings.to(dtype=input_embeddings.dtype)
         
        # -------------
        # Pass through llama 
        # -------------
        outputs_all = self.llama(inputs_embeds = input_embeddings, attention_mask=attention_mask, \
                             position_ids=position_ids, output_hidden_states=True, labels=None)
        
        # -------------
        # get logits and predicted audio tokens and pose embeddings
        # -------------
        # outputs_all.loss = scalar
        # outputs_all.logits = (B, max_words, 128256)
        # outputs_all.hidden_states (tuples) (B, max_words, 4096)
        ce_loss = outputs_all.loss
        out_logits = outputs_all.logits
        output_hidden_states = outputs_all.hidden_states[-1]
        
        modality_mask_pad_out = torch.zeros_like(modality_mask_pad, device=modality_mask_pad.device)
        # left shift the mask by one
        modality_mask_pad_out[:, :-1] = modality_mask_pad[:, 1:]
        target_mask_pad_out = torch.zeros_like(target_mask_pad, device=target_mask_pad.device)
        # left shift the mask by one
        target_mask_pad_out[:, :-1] = target_mask_pad[:, 1:]
        
        labels_out = torch.zeros_like(labels, device=labels.device)
        labels_out[:, :-1] = labels[:, 1:]
        
        # -------------
        # decode text
        # -------------
        tgt_text_ids_out = torch.where((labels_out != self.tokenizer.pose_token_id) & (labels_out != self.tokenizer.scue_token_id) & (labels_out != self.tokenizer.audio_token_id) & (target_mask_pad_out == 1))
            
        pred_text_logits = []
        target_gt_text = []    
        if len(tgt_text_ids_out[0]) != 0:
            pred_text_logits = out_logits[tgt_text_ids_out]
            target_gt_text = labels_out[tgt_text_ids_out]
        
        # -------------
        # decode audio
        # -------------
        pred_audio_logits = []
        tgt_audio_ids_out = torch.where((labels_out== self.tokenizer.audio_token_id) & (target_mask_pad_out == 1)) 
        assert torch.numel(tgt_audio_ids_out[0]) == torch.numel(tgt_audio_ids[0]), 'tgt_audio_ids/out should have the same number of elements'
    
        if len(tgt_audio_ids_out[0]) != 0:
            output_audio_embeddings = output_hidden_states[tgt_audio_ids_out]
            pred_audio_logits = self.decode_modality(output_audio_embeddings, modal='audio')
                
        # -------------
        # decode pose
        # -------------
        pred_pose = []
        output_pose_embeddings = []
        tgt_pose_ids_out = torch.where((labels_out== self.tokenizer.pose_token_id) & (target_mask_pad_out == 1))
        assert torch.numel(tgt_pose_ids_out[0]) == torch.numel(tgt_pose_ids[0]), 'tgt_pose_ids/out should have the same number of elements'
        if len(tgt_pose_ids_out[0]) != 0:
            output_pose_embeddings = output_hidden_states[tgt_pose_ids_out]
            pred_pose = self.decode_modality(output_pose_embeddings, modal='body')
                
        # -------------
        # predict speaking-state score
        # -------------
        pred_text_first_emb = []
        for labels_out_, target_mask_pad_out_, output_hidden_states_ in zip(labels_out, target_mask_pad_out, output_hidden_states):
            tgt_text_ids_out_ = torch.where((labels_out_ != self.tokenizer.pose_token_id) & (labels_out_ != self.tokenizer.scue_token_id) & (labels_out_ != self.tokenizer.audio_token_id) & (target_mask_pad_out_ == 1))
            pred_text_first_emb += [output_hidden_states_[tgt_text_ids_out_[0][0]]]
         
        pred_text_first_emb = torch.stack(pred_text_first_emb, dim=0)
        pred_speaking_state = self.pred_state(pred_text_first_emb)
        
        outputs = {'ce_loss':ce_loss, 'out_logits': out_logits,
                    'pred_text_logits': pred_text_logits, 'gt_text_ids': target_gt_text,
                    'pred_audio_logits': pred_audio_logits, 'gt_audio_ids': target_gt_audio_data,
                    'pred_pose': pred_pose, 'gt_pose': target_gt_pose_data,
                    'pred_speaking_state': pred_speaking_state}
        
        return outputs

    @torch.inference_mode()
    def get_inference_embeddings(self, input_tokens_pad, target_mask_pad, \
            obs_batch_audio_data, tgt_batch_audio_data, 
            obs_batch_pose_data, tgt_batch_pose_data, interact_batch_pose_data):
        
        # -------------
        # Text embeddings
        # -------------
        # B, 8192, 4096
        input_embeddings = self.llama.get_input_embeddings()(input_tokens_pad)
        #B, _, _ = input_embeddings.shape
        
        # -------------
        # Encode/insert audio embeddings
        # -------------
        if len(obs_batch_audio_data) != 0:
            obs_batch_audio_data = torch.cat(obs_batch_audio_data, dim=0)
            
            obs_audio_ids = torch.where((input_tokens_pad == self.tokenizer.audio_token_id) & (target_mask_pad != 1))     
            obs_audio_embeddings = self.encode_modality(obs_batch_audio_data, modal='audio')
            input_embeddings[obs_audio_ids] = obs_audio_embeddings.to(dtype=input_embeddings.dtype)
            
        if len(tgt_batch_audio_data) != 0:
            tgt_batch_audio_data = torch.cat(tgt_batch_audio_data, dim=0)
            
            tgt_audio_ids = torch.where((input_tokens_pad == self.tokenizer.audio_token_id) & (target_mask_pad == 1))     
            if len(tgt_audio_ids[0]) != 0:
                tgt_audio_embeddings = self.encode_modality(tgt_batch_audio_data, modal='audio')
                input_embeddings[tgt_audio_ids] = tgt_audio_embeddings.to(dtype=input_embeddings.dtype)
                         
        # -------------
        # Encode/insert pose embeddings
        # -------------
        
        # ---- observed pose embeddings ---- 
        if self.args.pose_fusion != 0:
            
            if len(obs_batch_pose_data) != 0:
                
                obs_batch_pose_data = torch.cat(obs_batch_pose_data, dim=0)
                
                obs_batch_pose_data = rearrange(obs_batch_pose_data, '(b t) d -> b t d', t = 5 * self.args.pose_hist_length)
                
                # find index for observation pose
                obs_pose_ids = torch.where((input_tokens_pad == self.tokenizer.pose_token_id) & (target_mask_pad != 1))
                obs_pose_embeddings = self.encode_modality(obs_batch_pose_data, modal='body_fusion')
                
                obs_pose_embeddings = rearrange(obs_pose_embeddings, 'b t d -> (b t) d')
                input_embeddings[obs_pose_ids] = obs_pose_embeddings.to(dtype=input_embeddings.dtype)
                
        else:
            if len(obs_batch_pose_data) != 0:
                # B5T, 327
                obs_batch_pose_data = torch.cat(obs_batch_pose_data, dim=0) # B, (5 * (T, L) )
                obs_pose_ids = torch.where((input_tokens_pad == self.tokenizer.pose_token_id) & (target_mask_pad != 1))
                obs_pose_embeddings = self.encode_modality(obs_batch_pose_data, modal='body')
                input_embeddings[obs_pose_ids] = obs_pose_embeddings.to(dtype=input_embeddings.dtype)
                
        if self.args.social_cue != 0:
            if len(interact_batch_pose_data) != 0:
                interact_feat = torch.stack(interact_batch_pose_data, dim=0)
            
            interact_embeddings = self.encode_modality(interact_feat, modal='social_cue')
            interact_ids = torch.where((input_tokens_pad == self.tokenizer.scue_token_id) & (target_mask_pad != 1))
            interact_embeddings = rearrange(interact_embeddings, 'b t d -> (b t) d')
            input_embeddings[interact_ids] = interact_embeddings.to(dtype=input_embeddings.dtype)    
                
        if len(tgt_batch_pose_data) != 0:
            tgt_batch_pose_data = torch.cat(tgt_batch_pose_data, dim=0)
            
            tgt_pose_ids = torch.where((input_tokens_pad == self.tokenizer.pose_token_id) & (target_mask_pad == 1))     
            tgt_pose_embeddings = self.encode_modality(tgt_batch_pose_data, modal='body')
            input_embeddings[tgt_pose_ids] = tgt_pose_embeddings.to(dtype=input_embeddings.dtype)
    
        return input_embeddings
        
    @torch.inference_mode()
    def forward_inference(self, input_embeddings, attention_mask = None, past_key_values = None, cache_position = None, position_ids=None, output_attentions =False):
        
        outputs_all = self.llama(inputs_embeds = input_embeddings, output_hidden_states=True
                                 , past_key_values=past_key_values
                                 , attention_mask=attention_mask
                                 , cache_position=cache_position
                                 , position_ids = position_ids, use_cache=True
                                 , output_attentions=output_attentions)
        
        return outputs_all

            
class PolySLGenWrapper(nn.Module):
    def __init__(self, args, model, tokenizer, logger=None):
        super().__init__()
        
        self.polyslgen = PolySLGen(args, model, tokenizer)
        self.loss_audio = nn.MSELoss()
        
        loss_weights = torch.ones(self.polyslgen.llama.config.vocab_size)
        self.cross_entropy_loss_text = torch.nn.CrossEntropyLoss(weight=loss_weights)
        self.bce_loss_turn = torch.nn.BCEWithLogitsLoss()
            
        self.max_words = args.max_words
    
        if args.eval_only != 0:
            self.pose_mean = np.load(f'{args.pretrained_model_dir}/Mean_c{args.chunk_length}_data_skeleton.npy')
            self.pose_std = np.load(f'{args.pretrained_model_dir}/Std_c{args.chunk_length}_data_skeleton.npy')
        else:
            self.pose_mean = np.load(f'{args.output_dir}/Mean_c{args.chunk_length}_data_skeleton.npy')
            self.pose_std = np.load(f'{args.output_dir}/Std_c{args.chunk_length}_data_skeleton.npy')
        
        self.pose_mean = nn.Parameter(torch.from_numpy(self.pose_mean).float(), requires_grad=False)
        self.pose_std = nn.Parameter(torch.from_numpy(self.pose_std).float(), requires_grad=False)
            
        self.args = args
        
        joint_init = np.load(f'{args.dnd_joint_init_path}', allow_pickle=True).flat[0]
        self.body_skeleton = Skeleton(joint_init)
                 
        # summarize the whole model
        for name, param in self.named_parameters():
            if 'polyslgen.llama' not in name and param.requires_grad:
               logger.info(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for name, p in self.named_parameters() if 'polyslgen.llama' not in name and p.requires_grad)
        logger.info(f"---------------- [non polyslgen.llama] Parameter count : {count} ----------------")
        
        for name, param in self.named_parameters():
            if 'polyslgen.llama' in name and param.requires_grad:
               logger.info(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for name, p in self.named_parameters() if 'polyslgen.llama' in name and p.requires_grad)
        logger.info(f"---------------- [polyslgen.llama] Parameter count : {count} ----------------")
    
    
    def forward(self, input_tokens_pad, target_mask_pad, modality_mask_pad, audio_data, pose_data, labels=None, is_train = True, speaking_state=None):
        
        bsz = len(input_tokens_pad)
        gt_batch_speaking_state = speaking_state
        
        output = self.polyslgen(input_tokens_pad, target_mask_pad, modality_mask_pad, audio_data, pose_data, labels=labels)
        
        ce_loss = torch.tensor(0.0).to(device=output['out_logits'].device)
        loss_audio = torch.tensor(0.0).to(device=ce_loss.device)
        loss_body = torch.tensor(0.0).to(device=ce_loss.device)
        loss_body_root_l2 = torch.tensor(0.0).to(device=ce_loss.device)
        loss_body_mpjpe = torch.tensor(0.0).to(device=ce_loss.device)
        reg_loss = torch.tensor(0.0).to(device=ce_loss.device)
        
        #### text ####
        if len(output['gt_text_ids']) != 0:
            ce_loss = self.cross_entropy_loss_text(output['pred_text_logits'].reshape(-1, output['pred_text_logits'].shape[-1]), output['gt_text_ids'].reshape(-1).long())
            
        #### speaking-state ####
        loss_spk_state = self.bce_loss_turn(output['pred_speaking_state'].squeeze(-1), gt_batch_speaking_state)
        
        #### audio ####   
        if len(output['gt_audio_ids']) != 0 and len(torch.where(gt_batch_speaking_state == 1)[0]) != 0:
            pred_audio_logits = output['pred_audio_logits'].reshape(bsz, 1, -1)
            pred_audio_logits = pred_audio_logits[gt_batch_speaking_state == 1]
            gt_audio_logits = output['gt_audio_ids'].reshape(bsz, 1, -1)
            gt_audio_logits = gt_audio_logits[gt_batch_speaking_state == 1]
            loss_audio = self.loss_audio(pred_audio_logits.reshape(-1, pred_audio_logits.shape[-1]), gt_audio_logits.reshape(-1, gt_audio_logits.shape[-1]))
        
        #### pose ####   
        if len(output['gt_pose']) != 0:
            # here we only calculate representation l2 loss
            gt_norm = (output['gt_pose'] - self.pose_mean) / self.pose_std
            loss_body = ((output['pred_pose'] - gt_norm) ** 2).sum(-1).mean()
            # recontruct body pose
            pred_pose_unnorm = output['pred_pose'] * self.pose_std + self.pose_mean
            gt_pose_unnorm = output['gt_pose']
        
            # reconstruct joint positions
            pred_wholebody_pose = self.body_skeleton.forward(pred_pose_unnorm)
            gt_wholebody_pose = self.body_skeleton.forward(gt_pose_unnorm)
                
            ############  physics-based loss ############# 
            # # https://github.com/Boeun-Kim/MoST/blob/main/model/loss.py
            
            # velocity regularization 
            pred_vel = pred_wholebody_pose.reshape(bsz, self.args.chunk_length, -1, 3)[:, 1:] - pred_wholebody_pose.reshape(bsz, self.args.chunk_length, -1, 3)[:, :-1]
            reg_vel = torch.mean((pred_vel).norm(dim=2))

            # acceleration regularization
            pred_acc = pred_vel[:, 1:] - pred_vel[:, -1:]
            reg_acc = torch.mean((pred_acc).norm(dim=2))

            # foot contact regularization
            gt_body = gt_wholebody_pose.reshape(bsz, self.args.chunk_length, -1, 3)
            
            toe_joints =  ['LeftFoot', 'LeftToeBase']
            fid_l_idx = torch.tensor([DND_JOINT_NAMES.index(i) for i in  toe_joints])
            
            toe_joints =  ['RightFoot', 'RightToeBase']
            fid_r_idx = torch.tensor([DND_JOINT_NAMES.index(i) for i in  toe_joints])
            
            fid_l, fid_r = torch.tensor(fid_l_idx).to(device=gt_body.device), torch.tensor(fid_r_idx).to(device=gt_body.device)
            velfactor = torch.tensor([0.05, 0.05]).to(device=gt_body.device)
            feet_contact = []
            for fid_index in [fid_l, fid_r]:
                foot_vel = (gt_body[:, 1:, fid_index] - gt_body[:, :-1, fid_index]) ** 2
                foot_vel = torch.sum(foot_vel, axis=-1)
                foot_contact = (foot_vel < velfactor).float()
                feet_contact.append(foot_contact)
            feet_contact = torch.cat(feet_contact, dim=-1)
            
            foot_idx = torch.cat((fid_l, fid_r), dim=-1)
            pred_foot = pred_wholebody_pose.reshape(bsz, self.args.chunk_length, -1, 3)[:, :, foot_idx]
            
            pred_foot_vel = pred_foot[:, 1:] - pred_foot[:, :-1]
            pred_foot_vel_sq = torch.norm(pred_foot_vel, dim=-1)
            pred_foot_vel_sq = pred_foot_vel_sq[feet_contact == 1]
            reg_contact = torch.sum(pred_foot_vel_sq)/len(pred_foot_vel_sq)
            
            reg_loss = reg_vel + 0.1 * reg_acc + reg_contact
            ##############################################
            
            pred_root = pred_wholebody_pose[:, :1]
            gt_root = gt_wholebody_pose[:, :1]
            
            loss_body_root_l2 = ((pred_root - gt_root) ** 2).sum(-1).mean()
              
            pred_wholebody_pose_centered = pred_wholebody_pose - pred_root
            gt_wholebody_pose_centered = gt_wholebody_pose - gt_root
            loss_body_mpjpe = ((pred_wholebody_pose_centered - gt_wholebody_pose_centered) ** 2).sum(-1).mean()
             
        
        #### validation ####  
        eval_body_root_l2 = torch.tensor(0.0).to(device=ce_loss.device)
        eval_body_mpjpe = torch.tensor(0.0).to(device=ce_loss.device)
        eval_text_tok_acc = torch.tensor(0.0).to(device=ce_loss.device)
          
        if is_train is False:
               
            #### text ####     
            if len(output['gt_text_ids']) != 0:
                pred_text_tok = torch.argmax(output['pred_text_logits'].softmax(dim=-1), dim=-1)
                eval_text_tok_acc = (pred_text_tok.reshape(-1) == output['gt_text_ids'].reshape(-1)).float().mean()
                
            #### pose ####  
            if len(output['gt_pose']) != 0:
                eval_body_root_l2 = torch.sqrt(((pred_root - gt_root) ** 2).sum(-1)).mean()
                eval_body_mpjpe = torch.sqrt(((pred_wholebody_pose_centered - gt_wholebody_pose_centered) ** 2).sum(-1)).mean()
          
            
        loss_all = ce_loss + loss_audio * self.args.loss_audio_weight \
                           + loss_body * self.args.loss_pose_weight + loss_body_mpjpe * self.args.loss_mpjpe_weight + loss_body_root_l2 * self.args.loss_l2_root_weight
        loss_all += reg_loss *  self.args.loss_reg_weight
        loss_all += loss_spk_state * self.args.loss_turn_weight
            
        losses = {'loss_all': loss_all, 'loss_text': ce_loss, 'loss_audio': loss_audio, 
                  'loss_body': loss_body,'loss_body_root_l2': loss_body_root_l2, 'loss_body_mpjpe': loss_body_mpjpe, 'loss_body_reg': reg_loss,
                   'loss_speaking_state': loss_spk_state
                   , 'eval_body_root_l2': eval_body_root_l2, 'eval_body_mpjpe': eval_body_mpjpe
                   , 'eval_text_tok_acc': eval_text_tok_acc}
                
        return losses
    
    @torch.inference_mode()
    def generate(self, prompt_tokens, prompt_embeddings, temperature = 0.6, top_k= 50, top_p= 0.9):
        
        bsz = len(prompt_tokens)
        
        min_prompt_len = max([len(i) for i in prompt_tokens])
        
        max_text_length = self.args.chunk_length
        max_audio_length = self.args.num_style_tokens
        max_pose_length = self.args.chunk_length
        
        total_len = min_prompt_len + max_text_length + max_audio_length + max_pose_length
        
        input_tokens_pad = torch.full((bsz, total_len), self.polyslgen.pad_token_id).cuda().long()
        input_embeddings_pad = torch.full((bsz, total_len, self.polyslgen.llama.config.hidden_size), 0).cuda().to(dtype=self.polyslgen.llama.dtype)
        
        speaking_state_out = torch.full((bsz, 1), -1).cuda().to(dtype=self.polyslgen.llama.dtype)
        audio_embeddings_out = torch.full((bsz, max_audio_length, self.args.style_dim), 0).cuda().to(dtype=self.polyslgen.llama.dtype)
        
        wholebody_pose_out = torch.full((bsz, max_pose_length, self.args.num_body_joints, 3), 0).cuda().float()
        pose_repre_out = torch.full((bsz, max_pose_length, self.polyslgen.pose_dim), 0).cuda().float()
        
        for k, (t, e) in enumerate(zip(prompt_tokens, prompt_embeddings)):
            input_tokens_pad[k, min_prompt_len-len(t):min_prompt_len] = t.long()
            input_embeddings_pad[k, min_prompt_len-len(e):min_prompt_len] = e
            
        prev_pos = 0
        past_key_values = DynamicCache()
            
        cur_text_length = torch.full((bsz, ), 0).long()
        is_text_finished = torch.full_like(cur_text_length, False)
        
        cur_audio_length = torch.full((bsz, ), 0).long()
        is_audio_finished = torch.full_like(cur_audio_length, False) 
            
        cur_pose_length = torch.full((bsz, ), 0).long()
        is_pose_finished = torch.full_like(cur_pose_length, False)
        
           
        # https://huggingface.co/docs/transformers/en/kv_cache
        for cur_pos in range(min_prompt_len, total_len):
            
            # get embeddings
            if cur_pos == min_prompt_len:
                cache_position = torch.ones_like(input_embeddings_pad[0, :cur_pos, 0], dtype=torch.int64).cumsum(0) - 1
                
            attention_mask = torch.where(input_tokens_pad == self.polyslgen.pad_token_id, 0, 1)  
            position_ids = attention_mask[:, :cur_pos].long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask[:, :cur_pos] == 0, 1)
                
            # clear cache
            outputs_all = self.polyslgen.forward_inference(input_embeddings_pad[:, prev_pos:cur_pos], 
                                                         past_key_values=past_key_values,
                                                         cache_position=cache_position,
                                                         attention_mask=attention_mask[:, :cur_pos],
                                                         position_ids=position_ids[:, prev_pos:cur_pos]
                                                         )
            
            cache_position = cache_position[-1:] + 1
        
            past_key_values = outputs_all.past_key_values
            
            # take the last one
            next_logits = outputs_all.logits[:, -1]
            next_embeddings = outputs_all.hidden_states[-1][:, -1]
            
            #### for each sample, track if each modality is finished ####
            text_b_ids = torch.where((is_text_finished == False))[0]
            audio_b_ids = torch.where((is_text_finished == True) & (is_audio_finished == False))[0]
            pose_b_ids = torch.where((is_text_finished == True) & (is_audio_finished == True) & (is_pose_finished == False))[0]
            
            #### text ####
            if len(text_b_ids) != 0:
                
                # pre-process distribution
                next_token_scores = self.logits_processor(next_logits[text_b_ids], top_k=top_k, top_p=top_p, temperature=temperature)
                
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # put token back to input_token_pad
                input_tokens_pad[text_b_ids, cur_pos] = next_token
                input_embeddings_pad[text_b_ids, cur_pos] = self.polyslgen.llama.get_input_embeddings()(next_token)
                cur_text_length[text_b_ids] += 1
                
                # check maximum length
                is_text_finished[torch.where(cur_text_length == max_text_length)] = True
                # check eot_id
                is_text_finished[torch.where(input_tokens_pad[:, cur_pos] == self.polyslgen.tokenizer.eos_token_id)] = True
                
                if cur_pos == min_prompt_len:
                    speaking_state_out[text_b_ids, 0]= self.polyslgen.pred_state(next_embeddings[text_b_ids]).sigmoid()
               
            #### audio (style) ####
            if len(audio_b_ids) != 0:
                
                # next_embeddings: B, 4096, next_audio: B, 256
                next_audio = self.polyslgen.decode_modality(next_embeddings[audio_b_ids], modal='audio')
                
                # collect output
                audio_embeddings_out[audio_b_ids, cur_audio_length[audio_b_ids]] = next_audio
                
                # next_audio: B, 256 --> audio_embeddings: B, 4096
                audio_embeddings = self.polyslgen.encode_modality(next_audio, modal='audio')
                input_embeddings_pad[audio_b_ids, cur_pos] = audio_embeddings.to(dtype=input_embeddings_pad.dtype)
                input_tokens_pad[audio_b_ids, cur_pos] = self.polyslgen.tokenizer.audio_token_id
                
                cur_audio_length[audio_b_ids] += 1
                
                # check maximum length
                is_audio_finished[torch.where(cur_audio_length == max_audio_length)] = True
            
            #### pose ####
            if len(pose_b_ids) != 0:
                
                # next_embeddings: B, 4096, next_pose: B, 327
                next_pose = self.polyslgen.decode_modality(next_embeddings[pose_b_ids], modal='body')
            
                next_pose_unnorm = next_pose * self.pose_std + self.pose_mean
                    
                pose_repre_out[pose_b_ids, cur_pose_length[pose_b_ids]] = next_pose_unnorm
                
                next_wholebody_pose = self.body_skeleton.forward(next_pose_unnorm)
                wholebody_pose_out[pose_b_ids, cur_pose_length[pose_b_ids]] = next_wholebody_pose
                
                # pred_pose: B, 327 --> pose_embeddings: B, 4096
                pose_embeddings = self.polyslgen.encode_modality(next_pose.flatten(start_dim=1), modal='body')
                input_embeddings_pad[pose_b_ids, cur_pos] = pose_embeddings.to(dtype=input_embeddings_pad.dtype)
                input_tokens_pad[pose_b_ids, cur_pos] = self.polyslgen.tokenizer.pose_token_id
                
                # put token back to input_token_pad
                cur_pose_length[pose_b_ids] += 1
                
                # check maximum length
                is_pose_finished[torch.where(cur_pose_length == max_pose_length)] = True
                    
            is_all_finished = len(torch.where((is_text_finished == False) | (is_audio_finished == False) | (is_pose_finished == False))[0]) == 0
            if is_all_finished is True:
                del outputs_all
                break
                
            prev_pos = cur_pos
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs_all
        del past_key_values
        # ============================================================ 
        input_tokens_pad_np = input_tokens_pad.detach().float().cpu().numpy()
        cur_text_length_np = cur_text_length.detach().long().cpu().numpy()
        # collect predictions
        #### text ####
        # pred_text_tokens: B, L-target
        pred_text_tokens_out = []
        start_pos = min_prompt_len
            
        for cur_text_length_, input_tokens_pad_ in zip(cur_text_length_np, input_tokens_pad_np):
            if cur_text_length_ == 0:
                pred_text_tokens_out.append([])
            else:
                pred_text_tokens_out.append(input_tokens_pad_[start_pos:start_pos+cur_text_length_])
            
        #### audio ####
        style_out = []
        audio_embeddings_np = audio_embeddings_out.detach().float().cpu().numpy()
        for audio_embeddings_ in audio_embeddings_np:
            style_out.append(audio_embeddings_)
        
        #### pose ####
        pred_wholebody_pose_out = []
        pred_pose_out = []
        
        wholebody_pose_out_np = wholebody_pose_out.detach().float().cpu().numpy()
        for next_wholebody_pose_ in wholebody_pose_out_np:
            pred_wholebody_pose_out.append(next_wholebody_pose_)
            
        pose_repre_out_np = pose_repre_out.detach().float().cpu().numpy()
        for next_pose_ in pose_repre_out_np:
            pred_pose_out.append(next_pose_)
                
        
        #### state ######
        pred_state_out = []
        speaking_states_np = speaking_state_out.detach().float().cpu().numpy()
        for speaking_states_ in speaking_states_np:
            pred_state_out.append(speaking_states_)
        
        return pred_text_tokens_out, style_out, pred_wholebody_pose_out, pred_state_out, pred_pose_out
    
    def generate_and_collect(self, 
                 input_tokens, target_mask, modality_mask, audio_data, pose_data, labels=None,
                 temperature = 0.6, top_k= 50, top_p= 0.9, speaking_states = None):
        
        # mask the labels with target_mask_pad, only caclulate loss for the target subject (non-history)
        bsz = len(input_tokens)
        
        # start from the first during
        start_index = [torch.where(t==1)[0][0] for t in target_mask]
        min_prompt_len = max([t for i, t in zip(input_tokens, start_index)])
        max_prompt_len_right = max([len(i)-t for i, t in zip(input_tokens, start_index)])
        total_len = min_prompt_len + max_prompt_len_right
        
        input_tokens_pad = torch.full((bsz, total_len), self.polyslgen.pad_token_id).cuda().long()
        target_mask_pad = torch.full((bsz, total_len), 0).cuda().long()
        modality_mask_pad = torch.full((bsz, total_len), -1).cuda().long()
        
        # different lengths for each sample in input_tokens
        for k, (t, s) in enumerate(zip(input_tokens, start_index)):
            input_tokens_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
        
        for k, (t, s) in enumerate(zip(target_mask, start_index)):
            target_mask_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
            
        for k, (t, s) in enumerate(zip(modality_mask, start_index)):
            modality_mask_pad[k, min_prompt_len-s:min_prompt_len-s+len(t)] = torch.tensor(t).long()
             
        obs_batch_audio_data = audio_data['obs_data']
        
        # only take real tgt audio data (for encoding)
        tgt_batch_audio_data = []
        for labels_, target_mask_pad_, tgt_data_ in zip(input_tokens, target_mask, audio_data['tgt_data']):
            tgt_audio_ids_ = torch.where((labels_ == self.polyslgen.tokenizer.audio_token_id) & (target_mask_pad_ == 1))
            if len(tgt_audio_ids_[0]) != 0:
                tgt_batch_audio_data += [tgt_data_]
        
        # take all gt_data, will filter out the dummmy ones when saving the final gt
        target_gt_audio_data = audio_data['gt_data']
        obs_batch_pose_data = pose_data['obs_data']
        tgt_batch_pose_data = pose_data['tgt_data']
        target_gt_pose_data = pose_data['gt_data']
        
        interact_batch_pose_data = []
        if self.args.social_cue != 0:
            interact_batch_pose_data = pose_data['interact_data']
        
        gt_batch_speaking_state = torch.stack(speaking_states, dim=0)
        
        gt_embeddings = self.polyslgen.get_inference_embeddings(input_tokens_pad, target_mask_pad, \
            obs_batch_audio_data, tgt_batch_audio_data, 
            obs_batch_pose_data, tgt_batch_pose_data, interact_batch_pose_data)
        
        ##########################
        start_index = [ torch.where(t==1)[0][0] for t in target_mask_pad]
        prompt_embeddings = [ e[:s] for e, s in zip(gt_embeddings, start_index)]
        prompt_tokens = [ e[:s] for e, s in zip(input_tokens_pad, start_index)]
        
        # text <eot> <style> <pose> <pose> ... <pose>
        pred_text_tok, pred_audio_embs, pred_wholebody_pose, pred_speaking_state, pred_pose_out = self.generate(prompt_tokens, prompt_embeddings, temperature = temperature, top_k= top_k, top_p= top_p)
            
        preds = {'text_tok': pred_text_tok, 'audio_emb': pred_audio_embs,'pose': pred_wholebody_pose, 'speaking_state': pred_speaking_state, 'pose_repr': pred_pose_out}
        
        # ============================================================ 
        # collect gt
        # have to do this for each sample since they are in different sizes
        if labels is not None:
            #### text ####
            gt_text_tokens_out = []
            
            for input_tokens_pad_, target_mask_pad_, modality_mask_pad_ in zip(input_tokens_pad, target_mask_pad, modality_mask_pad):
                tgt_ids = torch.where((target_mask_pad_ == 1) & (input_tokens_pad_ != self.polyslgen.tokenizer.pose_token_id) & (input_tokens_pad_ != self.polyslgen.tokenizer.scue_token_id) & (input_tokens_pad_ != self.polyslgen.tokenizer.audio_token_id))
                if len(tgt_ids[0]) != 0:
                    gt_text_tokens_out.append(input_tokens_pad_[tgt_ids].detach().float().cpu().numpy())
                else:
                    gt_text_tokens_out.append([])
            
            #### audio ####   
            gt_audio_tokens_out = []
            for target_gt_audio_data_, target_mask_pad_, input_tokens_pad_ in zip(target_gt_audio_data, target_mask_pad, input_tokens_pad):
                gt_audio_ids = torch.where((target_mask_pad_ == 1) & (input_tokens_pad_ == self.polyslgen.tokenizer.audio_token_id))
                if len(gt_audio_ids[0]) != 0:
                    gt_audio_tokens_out.append(target_gt_audio_data_.detach().float().cpu().numpy())
                else: 
                    gt_audio_tokens_out.append([])
            
            #### pose, needs to reconstruct wholebody pose ####
            gt_wholebody_pose_out = []
            gt_pose_out = []
            if len(target_gt_pose_data) != 0:
                for gt_pose_unnorm_tmp in target_gt_pose_data:
                    gt_pose_unnorm_ = gt_pose_unnorm_tmp
                        
                    gt_pose_out += [gt_pose_unnorm_.detach().float().cpu().numpy()]
                    
                    gt_wholebody_pose_ = self.body_skeleton.forward(gt_pose_unnorm_)
                    gt_wholebody_pose_out += [gt_wholebody_pose_.detach().float().cpu().numpy()]
                
            gt_speaking_state = [] 
            for speaking_state_ in gt_batch_speaking_state:
                gt_speaking_state += [speaking_state_.detach().float().cpu().numpy()]
                       
            # all gts
            gts = {'text_tok': gt_text_tokens_out,
                   'audio_emb': gt_audio_tokens_out, 
                   'pose': gt_wholebody_pose_out,
                   'speaking_state': gt_speaking_state,
                   'pose_repr': gt_pose_out
                   }
            
            return preds, gts
        else:
            return preds
    
    def logits_processor(self, logits, temperature, top_k, top_p):
            
        min_tokens_to_keep = 1
        filter_value = -float("Inf")
        
        ##### temperature
        logits = logits / temperature

        ##### top_k
        top_k = max(top_k, min_tokens_to_keep)
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

        ##### top_p
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
        
        return logits
             
    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    