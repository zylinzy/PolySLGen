from typing import Optional
import copy
import numpy as np
from torch.utils.data import Sampler, Dataset
import torch
import glob
from model.tokenizer import *
from data.dnd_config import *   

from data.dnd_config import *
from evaluator.utils.quaternion import *
from utils.rotation_conversions import rotation_6d_to_matrix

SIL_CAND = ['...', '......', '.........'
            '[listening...]', '[listening]', '(listening...)', '(listening)',  '<listening...>', '<listening>', 
            '[...listening...]', '[...listening]', '(...listening...)', '(...listening)', '<...listening...>', '<...listening>',
            '(processing...)', '(processing)', '[processing...]', '[processing]', '<processing...>', '<processing>',
            '(...processing...)', '(...processing)', '[...processing...]', '[...processing]', '<...processing...>', '<...processing>',
            '(thinking...)', '(thinking)', '[thinking...]', '[thinking]', '<thinking...>', '<thinking>',
            '(...thinking...)', '(...thinking)', '[...thinking...]', '[...thinking]', '<...thinking...>', '<...thinking>']

def make_interact_feat(args, obs_pose_data, body_skeleton):
        
    def get_score(body_skeleton, pose_other, pose_target):
        
        head_rot = body_skeleton.get_global_rot(pose_other)
        head_rot = head_rot[:, :,ROT_DND_JOINT_NAMES.index('Head')]
        
        head_pos = body_skeleton(pose_other)
        head_pos = head_pos[:, :, DND_JOINT_NAMES.index('Head')]
        
        head_pos_target = body_skeleton(pose_target)
        head_pos_target = head_pos_target[:, :, DND_JOINT_NAMES.index('Head')]
        
        # B, T, 3
        head_pos_relative = head_pos - head_pos_target
        
        # B, T, 3
        head_normal = torch.tensor([[[0, 0, -1]]]).float().repeat(head_pos_target.shape[0], head_pos_target.shape[1], 1).to(device=head_pos_relative.device)
        # B, T, 3, 3
        head_normal = (rotation_6d_to_matrix(head_rot) @ head_normal.unsqueeze(-1)).squeeze(-1)

        sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        scores = -sim(head_normal, head_pos_relative) # negative score
        
        return scores, head_pos, head_normal
    
    T = args.pose_hist_length  
    pose = []
    tot_num_participants = obs_pose_data.shape[1] // T
    for i in range(tot_num_participants):
        pose += [obs_pose_data[:, i*T:(i+1)*T]]
    
    other_pose = pose[:-1]
    target_pose = pose[-1]
    
    scores = []
    head_poss = []
    head_normals = []
    for other_pose_ in other_pose:
        score, head_pos, head_normal = get_score(body_skeleton, other_pose_, target_pose)
        
        scores += [score]
        head_poss += [head_pos]
        head_normals += [head_normal]
    
    final_feat = torch.stack(scores, dim=-1).unsqueeze(-1)
        
    return final_feat

def chunk2Input(args, final_data, tokenizer, target_persons, partition, is_baseline=False):
    
    EOS_TOKEN = tokenizer.eos_token
    POSE_TOKEN = tokenizer.pose_token
    AUDIO_TOKEN = tokenizer.audio_token
    SCUE_TOKEN = tokenizer.scue_token
    
    is_train = partition == 'train'
    chatFormatter = ChatFormat(tokenizer, args.chunk_length, target_persons[0])
    
    if args.social_cue != 0:
        joint_init = np.load(f'{args.dnd_joint_init_path}', allow_pickle=True).flat[0]
        
        from utils.dnd_skeleton import Skeleton, BvhJoint
        body_skeleton = Skeleton(joint_init)
        
    chunk_inputs = [] 
    chunk_tokens = []
    chunk_tokens_person = []
    chunk_target_mask = []
    chunk_audio_data = []
    chunk_pose_data = []
    chunk_modality_mask = []
    chunk_speaking_state = []
    
    # -------
    # For each target person, create its chunk data
    # -------
    # make the dataset balanced
    verbal_index = []
    nonverbal_index = []
    for i, chunk in enumerate(final_data):
        #print(chunk['conv_type'])
        if chunk['conv_type'] in ['h0c1', 'h1c1']:
            verbal_index.append(i)
        if chunk['conv_type'] in ['h0c0', 'h1c0']:
            nonverbal_index.append(i)
    
    num_samples = int(min(len(verbal_index), len(nonverbal_index)))
    nonverbal_index = nonverbal_index[:num_samples]

    target_index =  verbal_index + nonverbal_index
    
    target_person = target_persons[0]
    pose_all = []
    #for target_person in target_persons:
    for ch_i, chunk in enumerate(final_data):
        
        if ch_i not in target_index:
            continue
            
        dialog = {}
        dialog['content'] = []
        dialog['chunk_id'] = chunk['chunk_id']
        dialog['session_id'] = chunk['session_id']
        dialog['target_person'] = target_persons[0]
        dialog['conv_type'] = chunk['conv_type']
        
        # skip chunks with no textual history
        if chunk['conv_type'] in ['h0c0', 'h0c1']:
            continue
        
        # add system message
        id2num = {'a': 1, 'b': 2, 'j': 3, 'l': 4}
        to_add_sys = {'role': 'system', 'content': "You are an intelligent assistant that plays DnD game with the other four people, user1, user2, user3, and user4. In this game you act as the DnD DM. Please finish and react based on the on-going conversation. Make the conversation smooth and engaging."}
        to_add_sys['is_during_target'] = False
        
        dialog['content'].append(to_add_sys)
        
        # separate history and during
        history_message = []
        during_message = []
        for m_i, message in enumerate(chunk['content']):
            ctx_type_ = str.split(message[0], '_')[0]
            if ctx_type_ == 'history':
                history_message += [message]
            elif ctx_type_ == 'during':
                during_message += [message]
            else:
                assert False, 'should not happen'
        
        assert len(during_message) == 1, 'during_message should only have one row'
        
        # history
        if args.pose_fusion != 0:
            history_pose_all = []
         
        for _, message in enumerate(history_message):
            
            person_id = message[1]
            words = message[4].strip()
            audio_token = message[5]
            pose = message[-1]
            
            if len(words) != 0:
                
                # ------- Pose -------    
                role = 'assistant' if person_id == target_person else f'user{id2num[person_id]}'
                to_add = {'role': role, 'content': "", 'is_during_target': False} 
                
                new_words = f'{words}'
                to_add['content'] += new_words + EOS_TOKEN
                
                # ------- Audio -------  
                # check if there is audio and pose
                assert audio_token is not None, 'audio must exists'
                audio_speechtokens = np.tile(audio_token[None], (args.num_style_tokens, 1))
                
                assert audio_speechtokens.shape[0] == args.num_style_tokens, 'wrong audio token padding 0'
                assert audio_speechtokens.shape[1] == audio_token.shape[0], 'wrong audio token padding 1'
                
                # Audio...
                words_to_code = ''
                words_to_code += AUDIO_TOKEN * args.num_style_tokens
                to_add['content'] += words_to_code
                if 'obs_audio' in to_add.keys():
                    to_add['obs_audio'] = np.concatenate((to_add['obs_audio'], audio_speechtokens.copy()), axis=0)
                else:
                    to_add['obs_audio'] = audio_speechtokens.copy()
                        
                dialog['content'].append(to_add)
                    
            else: 
                assert pose is not None, 'if text is None, then pose must exist'
                
                # ------- Pose -------  
                if pose is not None:
                    
                    if args.pose_fusion != 0:
                        history_pose_all += [pose]
                    else:
                        new_words = ''
                        new_words += POSE_TOKEN*(pose.shape[0]-1)
                        role = '[pose] assistant' if person_id == target_person else f'[pose] user{id2num[person_id]}'
                        
                        to_add = {'role': role, 'content': "", 'is_during_target': False}  
                        words_to_code =  f'{new_words}'
                        to_add['content'] += words_to_code
                            
                        # T, J, 3
                        pose_raw_gt = pose[:-1]
                        if 'obs_pose' in to_add.keys():
                            to_add['obs_pose'] = np.concatenate((to_add['obs_pose'], pose_raw_gt.copy()), axis=0)
                        else:
                            to_add['obs_pose'] = pose_raw_gt.copy()
            
                        dialog['content'].append(to_add)
            
        # add all pose
        if args.pose_fusion != 0:
            
            assert len(history_pose_all) != 0, 'must have history pose'
            
            new_words = ''
            new_words += POSE_TOKEN*(history_pose_all[0].shape[0]-1)
                
            to_add = {'role': '[pose]', 'content': "", 'is_during_target': False}  
            words_to_code =  f'{new_words}'
            to_add['content'] += words_to_code
            
            for pose_ in history_pose_all:
                
                pose_raw_gt = pose_[:-1]
                
                if 'obs_pose' in to_add.keys():
                    to_add['obs_pose'] = np.concatenate((to_add['obs_pose'], pose_raw_gt.copy()), axis=0)
                else:
                    to_add['obs_pose'] = pose_raw_gt.copy()
                  
            dialog['content'].append(to_add)
            if args.social_cue != 0:
                to_add = {'role': '[interact]', 'content': "", 'is_during_target': False}  
                to_add['content'] += SCUE_TOKEN*args.social_cue_length
                dialog['content'].append(to_add)
        else: 
            if args.social_cue != 0:
                to_add = {'role': '[interact]', 'content': "", 'is_during_target': False}  
                to_add['content'] += SCUE_TOKEN*args.social_cue_length
                dialog['content'].append(to_add)
                         
        
        ############### DURING ###############
        # xxx <eos> <style> <pose><pose>..<pose>
        to_add = {'role': 'assistant', 'content': ''}
        to_add['is_during_target'] = True
        
        for _, message in enumerate(during_message):
            
            person_id = message[1]
            words = message[4].strip()
            audio_token = message[5]
            pose = message[-1]
            
            if len(words) != 0:
                
                # ------- Pose ------- 
                new_words = f' {words}'
                to_add['content'] += new_words + EOS_TOKEN
                
                # ------- Audio -------  
                # check if there is audio and pose
                assert audio_token is not None, 'audio must exists'
                audio_speechtokens = np.tile(audio_token[None], (args.num_style_tokens, 1))
                
                assert audio_speechtokens.shape[0] == args.num_style_tokens, 'wrong audio token padding 0'
                assert audio_speechtokens.shape[1] == audio_token.shape[0], 'wrong audio token padding 1'
                
                new_words = ''
                new_words += AUDIO_TOKEN * args.num_style_tokens
                to_add['content'] += new_words
                
                if 'tgt_audio' in to_add.keys():
                    to_add['tgt_audio'] = np.concatenate((to_add['tgt_audio'], audio_speechtokens.copy()), axis=0)
                else:
                    to_add['tgt_audio'] = audio_speechtokens.copy()
                    
                if 'gt_audio' in to_add.keys():
                    to_add['gt_audio'] = np.concatenate((to_add['gt_audio'], audio_speechtokens.copy()), axis=0)
                else:
                    to_add['gt_audio'] = audio_speechtokens.copy()
                
                # ------- Pose -------
                if pose is not None:
                    
                    new_words = ''
                    new_words += POSE_TOKEN*(pose.shape[0]-1)
                
                    words_to_code =  f'{new_words}'
                    to_add['content'] += words_to_code
                    
                    pose_raw_gt = pose[:-1]
                    if 'tgt_pose' in to_add.keys():
                        to_add['tgt_pose'] = np.concatenate((to_add['tgt_pose'], pose_raw_gt.copy()), axis=0)
                    else:
                        to_add['tgt_pose'] = pose_raw_gt.copy()
                    
                    # normalized    
                    if 'gt_pose' in to_add.keys():
                        to_add['gt_pose'] = np.concatenate((to_add['gt_pose'], pose_raw_gt.copy()), axis=0)
                    else:
                        to_add['gt_pose'] = pose_raw_gt.copy()
            
            else:
                assert pose is not None, 'if text is None, then pose must exists'
                # ------- Pose -------  
                new_words = np.random.choice(SIL_CAND, 1)[0] + EOS_TOKEN
                words_to_code =  f'{new_words}'
                to_add['content'] += words_to_code
                
                new_words = ''
                new_words += AUDIO_TOKEN * args.num_style_tokens
                words_to_code =  f'{new_words}'
                to_add['content'] += words_to_code
                
                audio_speechtokens = np.zeros((args.num_style_tokens, args.style_dim))
                if 'tgt_audio' in to_add.keys():
                    to_add['tgt_audio'] = np.concatenate((to_add['tgt_audio'], audio_speechtokens.copy()), axis=0)
                else:
                    to_add['tgt_audio'] = audio_speechtokens.copy()
                    
                if 'gt_audio' in to_add.keys():
                    to_add['gt_audio'] = np.concatenate((to_add['gt_audio'], audio_speechtokens.copy()), axis=0)
                else:
                    to_add['gt_audio'] = audio_speechtokens.copy()
                
                new_words = ''
                new_words += POSE_TOKEN*(pose.shape[0]-1)
                words_to_code =  f'{new_words}'
                to_add['content'] += words_to_code
                
                pose_raw_gt = pose[:-1]
                if 'tgt_pose' in to_add.keys():
                    to_add['tgt_pose'] = np.concatenate((to_add['tgt_pose'], pose_raw_gt.copy()), axis=0)
                else:
                    to_add['tgt_pose'] = pose_raw_gt.copy()
                
                # normalized    
                if 'gt_pose' in to_add.keys():
                    to_add['gt_pose'] = np.concatenate((to_add['gt_pose'], pose_raw_gt.copy()), axis=0)
                else:
                    to_add['gt_pose'] = pose_raw_gt.copy()
        
        dialog['content'].append(to_add)
        assert str.split(dialog['content'][-1]['role'], ' ')[-1] == 'assistant', 'the last message must be from the assistant!'

        # -------
        # One dialog is one chunk, with a set target person
        # ------- 
        tokens, tokens_person, audio_data, pose_data = chatFormatter.encode_multimodal_dialog(dialog['content'])
        
        assert (len(audio_data["obs_data"]) + len(audio_data["tgt_data"])) == (torch.tensor(tokens) == tokenizer.audio_token_id).sum(), 'number of tokens and length of audio_tokens must match'
            
        if len(pose_data) == 0:
            assert False, 'This should not happen since we will always have target person pose'
            
        if len(tokens) > args.max_words:
            assert False, f'input length exceed max_words {args.max_words}'
           
        chunk_inputs.append(dialog)
        
        # -------
        # Mark the tokens from the target person: #1 non-history, #2 history, #0 none
        # -------
        # during_pose, history_pose, during_target, history_target, during_all, history_all
        speaking_state = 0 if chunk['conv_type'] in ['h1c0', 'h0c0'] else 1
        
        target_mask = []
        modality_mask = []
        for role in tokens_person:
            
            if role == 'header':
                target_mask += [0]
                continue
            
            is_target = str.split(role, '_')[-1] == 'target'
            is_assistant = str.split(role, '_')[0] == 'assistant'
            if is_target:
                target_mask += [1]
            elif is_assistant:
                target_mask += [2]
            else:
                target_mask += [0]
                
        modality_mask = []
        for tok, role in zip(tokens, tokens_person):
            
            if role == 'header':
                modality_mask += [-1]
                continue
            
            if tok == tokenizer.audio_token_id:
                modality_mask += [1]
            elif tok == tokenizer.pose_token_id:
                modality_mask += [2]
            elif tok == tokenizer.scue_token_id:
                modality_mask += [5]
            else:
                modality_mask += [0]
        
        # -------
        # Mark the modality of the tokens: #1 audio, #2 pose, #0 text
        # -------
                
        chunk_tokens.append(tokens)
        chunk_tokens_person.append(tokens_person)
        chunk_target_mask.append(target_mask)
        chunk_audio_data.append(audio_data)
        chunk_pose_data.append(pose_data)
        chunk_modality_mask.append(modality_mask)
        chunk_speaking_state.append(speaking_state)
    
    for i, pose_data_ in enumerate(chunk_pose_data):   
        
        root_pose_init_xz = pose_data_['tgt_data'][:1, :3].copy() * np.array([[1, 0, 1]])
        
        if 'obs_data' in chunk_pose_data[i].keys():
            chunk_pose_data[i]['obs_data'][:, :3] = chunk_pose_data[i]['obs_data'][:, :3] - root_pose_init_xz
        chunk_pose_data[i]['tgt_data'][:, :3] = chunk_pose_data[i]['tgt_data'][:, :3] - root_pose_init_xz
        chunk_pose_data[i]['gt_data'][:, :3] = chunk_pose_data[i]['gt_data'][:, :3] - root_pose_init_xz
             
    if is_baseline is False:        
        if is_train is True:
            
            pose_all = []
            for pose_data_ in chunk_pose_data: 
                if 'obs_data' in pose_data_.keys():
                    pose_all += [pose_data_['obs_data']]
                pose_all += [pose_data_['tgt_data']]
                
            pose_all = np.concatenate(pose_all, axis=0)
            
            Mean = pose_all.mean(axis=0)
            Std = pose_all.std(axis=0)
            
            Std[0:3] = Std[0:3].mean() / 1.0 # global pos
            Std_body = Std[3:].reshape(-1, 6)
            
            # root pos
            Std_body[rot_dnd_body_idxs[:1]] = Std_body[rot_dnd_body_idxs[:1]].mean() / 1.0
            # body pos
            Std_body[rot_dnd_body_idxs[1:]] = Std_body[rot_dnd_body_idxs[1:]].mean() / 1.0
            # left hand pos
            Std_body[rot_dnd_left_hand_idxs[1:]] = Std_body[rot_dnd_left_hand_idxs[1:]].mean() / 1.0
            # right hand pos
            Std_body[rot_dnd_right_hand_idxs[1:]] = Std_body[rot_dnd_right_hand_idxs[1:]].mean() / 1.0
            
            Std = np.concatenate((Std[0:3], Std_body.reshape(-1)), axis=-1)
            
            np.save(f'{args.output_dir}/Mean_c{args.chunk_length}_data_skeleton.npy', Mean)
            np.save(f'{args.output_dir}/Std_c{args.chunk_length}_data_skeleton.npy', Std)
        
            pose_mean = Mean
            pose_std = Std
        else:
            if args.eval_only != 0:
                pose_mean = np.load(f'{args.pretrained_model_dir}/Mean_c{args.chunk_length}_data_skeleton.npy')
                pose_std = np.load(f'{args.pretrained_model_dir}/Std_c{args.chunk_length}_data_skeleton.npy')
            else:
                pose_mean = np.load(f'{args.output_dir}/Mean_c{args.chunk_length}_data_skeleton.npy')
                pose_std = np.load(f'{args.output_dir}/Std_c{args.chunk_length}_data_skeleton.npy')
        
        for i, pose_data in enumerate(chunk_pose_data):
            if 'obs_data' in chunk_pose_data[i].keys():
                if args.social_cue != 0:
                    chunk_pose_data[i]['interact_data'] = make_interact_feat(args, torch.from_numpy(chunk_pose_data[i]['obs_data']).unsqueeze(0), body_skeleton)[0].numpy()
                        
                    assert chunk_pose_data[i]['interact_data'].shape[1] == 4, 'should have shape (t, 4, 1)'
                
                chunk_pose_data[i]['obs_data'] = (chunk_pose_data[i]['obs_data'] - pose_mean) / pose_std
                
                assert chunk_pose_data[i]['obs_data'].shape[0] == 5 * args.pose_hist_length, 'should have shape (5 * args.pose_hist_length, )'
                
            chunk_pose_data[i]['tgt_data'] = (chunk_pose_data[i]['tgt_data'] - pose_mean) / pose_std

    return {'raw_dialog': chunk_inputs, 'input_tokens': chunk_tokens, 'token_roles': chunk_tokens_person, \
            'token_role_mask': chunk_target_mask, 'modality_mask': chunk_modality_mask, 'audio_data': chunk_audio_data, 'pose_data': chunk_pose_data, 'speaking_state': chunk_speaking_state}
    
def preprocess_data(args, tokenizer, partition, logger=None, is_baseline=False):
    
    target_person = 'c'
    
    if partition == 'train' or partition == 'val':
        input_filenames = glob.glob(f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/Session_1_hp{args.pose_hist_length}_unsync.npy')
        input_filenames += glob.glob(f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/Session_2_hp{args.pose_hist_length}_unsync.npy')
        input_filenames += glob.glob(f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/Session_3_hp{args.pose_hist_length}_unsync.npy')
    
    elif partition == 'test':
        input_filenames = glob.glob(f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/Session_4_hp{args.pose_hist_length}_unsync.npy')

    datas_in = [] 
    logger.info('loading transcripts ...')
    for input_filename in input_filenames:
        logger.info(input_filename)
        data = np.load(input_filename, allow_pickle=True)
        
        datas_in.extend(data)
    
    index_path = f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/train_data_idx.npy'
    if partition != 'test':
        if os.path.exists(index_path) is False:
            import random
            index_all = list(range(len(datas_in)))
            random.shuffle(index_all)
            np.save(index_path, index_all)
        else:
            index_all = np.load(index_path, allow_pickle=True)
        
        datas = [datas_in[i] for i in index_all]
    else:
        datas = datas_in
        
    if partition == 'train':
        size_train = int(0.9 * len(datas) * args.train_ratio)
        datas = datas[:size_train]
    elif partition == 'val':
        size_val = int(0.1 * len(datas) * args.val_ratio)
        datas = datas[-size_val:]
    elif partition == 'test':
        size_test = int(len(datas) * args.test_ratio)
        datas = datas[:size_test]
                    
    logger.info(f'Finish loading {partition} transcripts ... size = {len(datas)}')
    
    target_persons = ['c']
    
    chunk_processed_data = chunk2Input(args, datas, tokenizer, target_persons, partition, is_baseline=is_baseline)
    
    data_length = len(chunk_processed_data['raw_dialog'])
    logger.info(f'Finish preprocessing {partition} data ... {data_length}')
    return chunk_processed_data
    
class DnDDataset(Dataset):
    def __init__(self, args, chunk_processed_data, tokenizer, partition='train'):
        
        self.client = None
        self.partition = partition
        self.max_words = args.max_words
        self.is_train = partition != 'test'
        self.pad_token_id = tokenizer.pad_token_id
        self.audio_token_id = tokenizer.audio_token_id
        self.pose_token_id = tokenizer.pose_token_id
        
        self.target_persons = ['c']
        self.args = args
        
        # -------
        # from chunk utterances to text string
        # -------
        self.chunk_processed_data = chunk_processed_data
        self.tot_num_chunk = len(self.chunk_processed_data['input_tokens'])
        self.num_chunk_per_person = self.tot_num_chunk // len(self.target_persons)
        
        self.group_indices = {}
        self.group_indices['verbal'] = []
        self.group_indices['nonverbal'] = []
        
        for i, speaking_state in enumerate(chunk_processed_data['speaking_state']):
            if speaking_state == 1:
                self.group_indices['verbal'] += [i]
            else:
                self.group_indices['nonverbal'] += [i]
            
    def __len__(self):
        return int(self.tot_num_chunk)
    
    def __getitem__(self, index):
        
        input_tokens = torch.tensor(self.chunk_processed_data['input_tokens'][index]).long()
        raw_dialog = self.chunk_processed_data['raw_dialog'][index]
        modality_mask = torch.tensor(self.chunk_processed_data['modality_mask'][index]).long()
        token_role_mask = torch.tensor(self.chunk_processed_data['token_role_mask'][index]).long()
        
        if self.is_train is True:
            padding = self.max_words - input_tokens.shape[0]
            if padding > 0:
                input_tokens_pad = torch.cat((self.pad_token_id * torch.ones(padding).long(), input_tokens))
                target_mask_pad = torch.cat((self.pad_token_id * torch.ones(padding).long(), token_role_mask))
                modality_mask_pad = torch.cat((self.pad_token_id * torch.ones(padding).long(), modality_mask))
            else:
                input_tokens_pad = input_tokens[-self.max_words:]
                target_mask_pad = token_role_mask[-self.max_words:]
                modality_mask_pad = modality_mask[-self.max_words:]
        else:
            padding = self.max_words - input_tokens.shape[0]
            if padding < 0:
                input_tokens_pad = input_tokens[-self.max_words:]
                target_mask_pad = token_role_mask[-self.max_words:]
                modality_mask_pad = modality_mask[-self.max_words:]
            else:
                padding = 0
                input_tokens_pad = input_tokens
                target_mask_pad = token_role_mask
                modality_mask_pad = modality_mask
          
        audio_data = {}     
        audio_data_in = self.chunk_processed_data['audio_data'][index]
        for k, v in audio_data_in.items():
            audio_data[k] = torch.from_numpy(v).float()
            if padding < 0:
                num_lost = (input_tokens[:-self.max_words] == self.audio_token_id).sum().int()
                audio_data[k] = audio_data[k][num_lost:]
        
        assert 'gt_data' in audio_data.keys(), 'audio_data is empty'
        
        pose_data = {}
        pose_data_in = self.chunk_processed_data['pose_data'][index]
        for k, v in pose_data_in.items():
            pose_data[k] = torch.from_numpy(v).float()
            if padding < 0:
                num_lost = (input_tokens[:-self.max_words] == self.pose_token_id).sum().int()
                pose_data[k] = pose_data[k][num_lost:]
                
        assert 'gt_data' in pose_data.keys(), 'pose_data is empty'
              
        labels = copy.deepcopy(input_tokens_pad)
        
        speaking_state = torch.tensor(self.chunk_processed_data['speaking_state'][index]).float()
        
        return raw_dialog, input_tokens_pad, labels, target_mask_pad, \
            modality_mask_pad, audio_data, pose_data, speaking_state
 
    def groups(self):
        return list(self.group_indices.values())    

class FinetuneDistSampler(Sampler):
    #   Distrubuted Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0
        self.start_iter = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_)//batch_size for _ in group_indices]
        assert all([_%num_replicas==0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)
        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        global_batch_size = self.batch_size * self.num_replicas * self.acc_grad
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            global_batched_indices = [
                indices_in_group[i:i+global_batch_size]
                for indices_in_group in group_indices_shuffle
                for i in range(0, len(indices_in_group), global_batch_size)]
            rng.shuffle(global_batched_indices)
            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = [_ for batch_indices in group_indices for _ in batch_indices]

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size:]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter

