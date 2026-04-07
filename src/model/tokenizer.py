# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import numpy as np

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    text: str
    audio: str
    pose: str

Dialog = Sequence[Message]
   



class ChatFormat:
    def __init__(self, tokenizer: AutoTokenizer, chunk_length, target_person):
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length
        self.audio_length = chunk_length * 2
        self.target_person = target_person
        

    def encode_multimodal_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.extend(self.tokenizer.encode("<|start_header_id|>", add_special_tokens = False))
        tokens.extend(self.tokenizer.encode(message["role"], add_special_tokens = False))
        tokens.extend(self.tokenizer.encode("<|end_header_id|>", add_special_tokens = False))
        tokens.extend(self.tokenizer.encode("\n\n", add_special_tokens = False))
        return tokens

    def encode_multimodal_dialog(self, dialog: Dialog):
        tokens = []
        tokens_person = []
        audio_data_dialog = {}
        pose_data_dialog = {}
        tokens.extend(self.tokenizer.encode("<|begin_of_text|>", add_special_tokens = False))
        tokens_person.extend(['header'])
        for i, message in enumerate(dialog):
            tok, tok_per, audio_data, pose_data = self.encode_multimodal_message(message)
            tokens.extend(tok)
            tokens_person.extend(tok_per)
            
            for k, v in audio_data.items():
                if k in audio_data_dialog.keys():
                    audio_data_dialog[k] = np.concatenate((audio_data_dialog[k], v), axis=0)
                else:
                    audio_data_dialog[k] = v
             
            for k, v in pose_data.items():
                if k in pose_data_dialog.keys():
                    pose_data_dialog[k] = np.concatenate((pose_data_dialog[k], v), axis=0)
                else:
                    pose_data_dialog[k] = v
                
        return tokens, tokens_person, audio_data_dialog, pose_data_dialog
    
    def encode_multimodal_message(self, message):
        
        tokens = self.encode_multimodal_header(message)
        n_header = len(tokens)
        tokens_person = ['header' for i in range(n_header)]
        
          
        start = len(tokens)
        tokens.extend(self.tokenizer.encode(message['content'].strip(), add_special_tokens = False))
        end = len(tokens)
        
        if 'is_during_target' in message.keys():
            post_ = '_target' if message['is_during_target'] is True else '_nontarget'
            role_ = str.split(message['role'], ' ')[-1]
            tokens_person += [role_+post_ for i in range(end-start)]
        else:
            role_ = str.split(message['role'], ' ')[-1]
            tokens_person += [role_ for i in range(end-start)]
        
        audio_data = {}
        if 'obs_audio' in message.keys():
            audio_data['obs_data'] = message["obs_audio"]
            
        if 'tgt_audio' in message.keys():
            audio_data['tgt_data'] = message["tgt_audio"]
            
        if 'gt_audio' in message.keys():
            audio_data['gt_data'] = message["gt_audio"]
            
        
        pose_data = {}  
        if 'obs_pose' in message.keys():
            pose_data['obs_data'] = message["obs_pose"]
                
        if 'tgt_pose' in message.keys():
            pose_data['tgt_data'] = message["tgt_pose"]
            
        if 'gt_pose' in message.keys():
            pose_data['gt_data'] = message["gt_pose"]
                
        return tokens, tokens_person, audio_data, pose_data
    