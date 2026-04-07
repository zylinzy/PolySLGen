import os
import tqdm
import numpy as np
import torch

from torcheval.metrics import BinaryAUPRC
import glob
from data.dnd_dataset import SIL_CAND

def calculate_eval_metrics(args, tokenizer, state_from_text = False, baseline_mode = None):
    
    output_dir = args.output_dir if baseline_mode is None else os.path.join(args.output_dir, baseline_mode)
    
    #### Speaking-state and text
    for ii in range(args.test_iter): 
        
        if state_from_text:
            get_state_metrics_from_text(output_dir, ii, tokenizer)
        else:
            get_state_metrics(output_dir, ii)
            
        get_text_metrics(tokenizer, output_dir, ii)
    
    # statistics for speaking-state
    if state_from_text:
        log_files = glob.glob(f'{output_dir}/test_state_metrics_from_text_*.txt')
        out_statistics_dict = get_statistics(log_files)
        dirname = f'{output_dir}/statistics/'
        os.makedirs(dirname, exist_ok=True)
        with open(f'{dirname}/test_state_metrics_from_text.txt', 'w') as f:
            json.dump(out_statistics_dict, f, indent=2)
    else:
        log_files = glob.glob(f'{output_dir}/test_state_metrics_from_flag_*.txt')
        out_statistics_dict = get_statistics(log_files)
        dirname = f'{output_dir}/statistics/'
        os.makedirs(dirname, exist_ok=True)
        with open(f'{dirname}/test_state_metrics_from_flag.txt', 'w') as f:
            json.dump(out_statistics_dict, f, indent=2)
       
    # statistics for text 
    log_files = glob.glob(f'{output_dir}/test_text_metrics_*.txt')
    out_statistics_dict = get_statistics(log_files)
    dirname = f'{output_dir}/statistics/'
    os.makedirs(dirname, exist_ok=True)
    with open(f'{dirname}/test_text_metrics.txt', 'w') as f:
        json.dump(out_statistics_dict, f, indent=2)
    
    #### Pose            
    for ii in range(args.test_iter):
        get_pose_metrics(args, output_dir, ii)
    
    # statistics for pose
    log_files = glob.glob(f'{output_dir}/test_pose_metrics_*.txt')
    out_statistics_dict = get_statistics(log_files)
    dirname = f'{output_dir}/statistics/'
    os.makedirs(dirname, exist_ok=True)
    with open(f'{dirname}/test_pose_metrics.txt', 'w') as f:
        json.dump(out_statistics_dict, f, indent=2)
      
    #### Speech  
    for ii in range(args.test_iter):    
        synthesize_speech_and_evaluate(output_dir, tokenizer, args.styletts_path, ii)
            
    # statistics for speech
    log_files = glob.glob(f'{output_dir}/test_speech_metrics_*.txt')
    out_statistics_dict = get_statistics(log_files)
    dirname = f'{output_dir}/statistics/'
    os.makedirs(dirname, exist_ok=True)
    with open(f'{dirname}/test_speech_metrics.txt', 'w') as f:
        json.dump(out_statistics_dict, f, indent=2)
            
def get_statistics(log_files):
    
    data_all = []
    for file in log_files:
        f = open(file)
        data_all += [json.load(f)]
        f.close()
    
    # creat empty dict
    out = {}
    for k in data_all[0].keys():
        out[k] = []
    
    # collect all metrics
    for data in data_all:
        for k, v in data.items():
            out[k] += [v]
            
    # compute
    final_out = {}
    for k, v in out.items():
        mean_ = np.array(v).astype(np.float32).mean().item()
        std_ = np.array(v).astype(np.float32).std().item()
        final_out[k] = "{:.5f} / {:.5f}".format(mean_, std_)
        
    return final_out  

def get_state_metrics(tgt_dir, iter):
    
    files = glob.glob(f'{tgt_dir}/*_results_{iter}.npy')
    aup_metric = BinaryAUPRC()
    
    predictions = []
    references = []
    for file in files:
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        if os.path.exists(f'{dirname}/{set_name}_state_metrics_from_flag_{iter}.txt') is True:
            continue
        
        data = torch.load(file, map_location="cpu", weights_only=False)
        
        for data_ in data:
            pred_states = data_['preds']['speaking_state']
            pred_states = np.stack(pred_states, axis=0).squeeze(-1)
            gt_states = data_['gts']['speaking_state']
            gt_states = np.stack(gt_states, axis=0)
              
            predictions += [pred_states]  
            references += [gt_states]
        
        predictions = np.concatenate(predictions, axis=0)        
        references = np.concatenate(references, axis=0).astype(np.int64)
        
        aup_metric.update(torch.from_numpy(predictions), torch.from_numpy(references))
        aup = aup_metric.compute().numpy()
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        with open(f'{dirname}/{set_name}_state_metrics_from_flag_{iter}.txt', 'w') as f:
            json.dump({'aup':aup.item()}, f, indent=2)

def get_state_metrics_from_text(tgt_dir, iter, tokenizer):
    
    files = glob.glob(f'{tgt_dir}/*_results_{iter}.npy')
                               
    aup_metric = BinaryAUPRC()
    
    for file in files:
        predictions = []
        references = []
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        if os.path.exists(f'{dirname}/{set_name}_state_metrics_from_text_{iter}.txt') is True:
            continue
        
        data = torch.load(file, map_location="cpu", weights_only=False)
        
        for data_ in data:
            pred_text_tokens = data_['preds']['text_tok']
            pred_states = []
            for i, gt_ in enumerate(pred_text_tokens):    
                pred_ = pred_text_tokens[i]
                pred_first_idx = len(pred_)
                if pred_first_idx == 0:
                    pred_states += [0]
                else: 
                    if (pred_ == 128001).any() | (pred_ == 128008).any() | (pred_ == 128009).any():
                        pred_first_idx = np.where((pred_ == 128001) | (pred_ == 128008) | (pred_ == 128009))[0][0]
                    
                    pred_text = tokenizer.decode(pred_[:pred_first_idx].astype(np.int32), skip_special_tokens=True)
                    if pred_text in SIL_CAND:
                        pred_states += [0]
                    else:
                        pred_states += [1]
                
            pred_states = np.stack(pred_states, axis=0)
            gt_states = []
            for raw_dialogue in data_['raw_dialogue']:
                gt_states += [int(raw_dialogue['conv_type'] in ['h1c1', 'h0c1'])]
                
            gt_states = np.stack(gt_states, axis=0)
              
            predictions += [pred_states]  
            references += [gt_states]  
        
        predictions = np.concatenate(predictions, axis=0)        
        references = np.concatenate(references, axis=0).astype(np.int64)
        
        aup_metric.update(torch.from_numpy(predictions), torch.from_numpy(references))
        aup = aup_metric.compute().numpy()
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        with open(f'{dirname}/{set_name}_state_metrics_from_text_{iter}.txt', 'w') as f:
            json.dump({'aup':aup.item()}, f, indent=2)
            
            
from evaluate import load
def get_text_metrics(tokenizer, tgt_dir, iter):
    
    EOS_TOKEN_ID = tokenizer.eos_token_id
    
    from bert_score import BERTScorer
    bertscore = BERTScorer(model_type='bert-large-uncased')
    wer = load("wer")
    files = glob.glob(f'{tgt_dir}/*_results_{iter}.npy')

    for file in files:
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        predictions = []
        references = []
        
        data = torch.load(file, map_location="cpu", weights_only=False)
        
        for batch_i, data_ in enumerate(tqdm(data)):
            
            pred_text_tokens = data_['preds']['text_tok']
            gt_text_tokens = data_['gts']['text_tok']
            
            batch_size = len(gt_text_tokens)
            data[batch_i]['preds']['text'] = [[] for i in range(batch_size)]
            data[batch_i]['gts']['text'] = [[] for i in range(batch_size)]
            
            data[batch_i]['preds']['bert'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['wer'] = [[] for i in range(batch_size)]
            
            for i, gt_ in enumerate(gt_text_tokens):
                
                if len(gt_) == 0:
                    continue
                
                pred_ = pred_text_tokens[i]
                
                gt_first_idx = len(gt_)
                #if (gt_ == 128001).any() | (gt_ == 128008).any() | (gt_ == 128009).any():
                #    gt_first_idx = np.where((gt_ == 128001) | (gt_ == 128008) | (gt_ == 128009))[0][0]
                if (gt_ == EOS_TOKEN_ID).any():
                    gt_first_idx = np.where(gt_ == EOS_TOKEN_ID)[0][0]
                
                pred_first_idx = len(pred_)
                if pred_first_idx == 0:
                    decoded_pred_text = ""
                    predictions += [decoded_pred_text]
                else: 
                    #if (pred_ == 128001).any() | (pred_ == 128008).any() | (pred_ == 128009).any():
                    #    pred_first_idx = np.where((pred_ == 128001) | (pred_ == 128008) | (pred_ == 128009))[0][0]
                    if (pred_ == EOS_TOKEN_ID).any():
                        pred_first_idx = np.where(pred_ == EOS_TOKEN_ID)[0][0]
                    
                    decoded_pred_text = tokenizer.decode(pred_[:pred_first_idx].astype(np.int32), skip_special_tokens=True)
                    predictions += [decoded_pred_text]
                
                decoded_gt_text = tokenizer.decode(gt_[:gt_first_idx].astype(np.int32), skip_special_tokens=True)
                references += [decoded_gt_text]
                
                data[batch_i]['gts']['text'][i] = decoded_gt_text
                
                data[batch_i]['preds']['text'][i] = decoded_pred_text
                
                #############
                predictions_tmp = [data[batch_i]['preds']['text'][i]]
                references_tmp = [data[batch_i]['gts']['text'][i]]
                
                _, _, text_f1_tmp = bertscore.score(predictions_tmp, references_tmp)
                data[batch_i]['preds']['bert'][i] = np.array(text_f1_tmp).mean().item()
                
                results_tmp = wer.compute(predictions=predictions_tmp, references=references_tmp)
                data[batch_i]['preds']['wer'][i] = np.array(results_tmp).item()
                ############
        
        torch.save(data, file)
            
        if len(references) != 0:        
            _, _, text_f1 = bertscore.score(predictions, references)
            text_f1_all = np.array(text_f1).mean()
            
            results = wer.compute(predictions=predictions, references=references)
            text_wer_all = np.array(results)
        else:
            text_f1_all = np.array([-1])
            text_wer_all = np.array([-1])
            
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        with open(f'{dirname}/{set_name}_text_metrics_{iter}.txt', 'w') as f:
            json.dump({'f1': text_f1_all.item(), 'wer': text_wer_all.item()}, f, indent=2)
            
    del bertscore
    del wer
 
from evaluator.network import MovementConvEncoder
from evaluator.utils.motion_process import process_motion, decompose
from utils.eval_similarity import *
from utils.rotation_conversions import rotation_6d_to_matrix
from utils.dnd_skeleton import Skeleton, BvhJoint
def get_pose_metrics(args, tgt_dir, iter):
    
    
    joint_init = np.load(f'{args.dnd_joint_init_path}', allow_pickle=True).flat[0]
    body_skeleton = Skeleton(joint_init)
    body_skeleton.cuda()
    
    # --- get data_raw
    target_person = 'c'
    input_filenames = glob.glob(f'{args.data_dir}/forecast_c{args.chunk_length}_h{args.hist_length}_p{target_person}/Session_4_hp{args.pose_hist_length}_unsync.npy')

    data_raw = [] 
    for input_filename in input_filenames:
        data = np.load(input_filename, allow_pickle=True)
        data_raw.extend(data)
    
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

    files = glob.glob(f'{tgt_dir}/*_results_{iter}.npy')
    
    def get_pose_from_raw_data(chunk_index, session_id, data_raw, p):
    
        for data_ in data_raw:
            if int(data_['chunk_id']) == chunk_index and data_['session_id'] == session_id:
                pose_all = []
                for i in range(p):
                    if i == p-1:
                        continue
                    
                    pose_all += [data_['content'][-1-p+i][-1][:-1]]
                    
                pose_target = data_['content'][-1][-1][:-1]
                
                return pose_all, pose_target
       
       
    def get_ang_diff(pred_rots, gt_rots):
        
        head_rot = body_skeleton.get_global_rot(torch.from_numpy(pred_rots).cuda())
        head_rot = head_rot[ :,ROT_DND_JOINT_NAMES.index('Head')]
        
        head_rot_target = body_skeleton.get_global_rot(torch.from_numpy(gt_rots).cuda())
        head_rot_target = head_rot_target[ :,ROT_DND_JOINT_NAMES.index('Head')]
        
        head_normal = torch.tensor([[0, 0, -1]]).float().to(device=head_rot.device)
        head_normal = (rotation_6d_to_matrix(head_rot) @ head_normal.unsqueeze(-1)).squeeze(-1)

        head_normal_target = torch.tensor([[0, 0, -1]]).float().to(device=head_rot_target.device)
        head_normal_target = (rotation_6d_to_matrix(head_rot_target) @ head_normal_target.unsqueeze(-1)).squeeze(-1)

        angles_deg = angular_difference(head_normal, head_normal_target)
        
        return angles_deg
             
    def get_att_score(pred_rots, tgt_pose, head_pos_obs):
        
        head_pos_target = body_skeleton(torch.from_numpy(pred_rots).cuda())
        head_pos_target = head_pos_target[ :, DND_JOINT_NAMES.index('Head')]
        
        # shift back to gt position (was centered to target)
        root_pose_init_xz = torch.from_numpy(tgt_pose[:1, :3].copy() * np.array([[1, 0, 1]])).cuda()
        head_pos_target += root_pose_init_xz
        
        head_rot_target = body_skeleton.get_global_rot(torch.from_numpy(pred_rots).cuda())
        head_rot_target = head_rot_target[:, ROT_DND_JOINT_NAMES.index('Head')]
        
        head_normal_target = torch.tensor([[0, 0, -1]]).float().to(device=head_pos_target.device)
        head_normal_target = (rotation_6d_to_matrix(head_rot_target) @ head_normal_target.unsqueeze(-1)).squeeze(-1)

        head_pos_relative = head_pos_target - head_pos_obs
        
        sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        score = -sim(head_normal_target, head_pos_relative)
        
        return score
        
    def to_motion(pred_, mean, std, feat_dim):
        
        data_unnorm = torch.from_numpy(process_motion(pred_))
        g_rot, motion = decompose(data_unnorm)
        motion = ((motion - mean) / std).reshape(-1, feat_dim-6)
        pred_motion = torch.cat((g_rot, motion), dim=-1) # T' K
        
        return pred_motion
    
    for file in files:
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        if os.path.exists(f'{dirname}/{set_name}_pose_metrics_all_{iter}.txt') is True:
            continue
        
        data = torch.load(file, map_location="cpu", weights_only=False)
        
        pred_motion_emb_all = []
        gt_motion_emb_all = []
        pred_motion_all = []
        gt_motion_all = []
        pred_all = []
        gt_all = []
        
        angle_mae_all = []
        diff_score_all = [[], [], [], []]
        
        for batch_i, data_ in enumerate(tqdm(data)):
            
            batch_size = len(data_['preds']['pose'])
            data[batch_i]['preds']['mpjpe'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['root'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['fid'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['head_mae'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['att_score_diff'] = [[[] for j in range(4)] for i in range(batch_size)]
            
            pred_motion_batch = []
            gt_motion_batch = []
            pred_motion_batch_enc = []
            gt_motion_batch_enc = []
            pred_batch = []
            gt_batch = []
            
            for i, (pred_, gt_) in enumerate(zip(data_['preds']['pose'], data_['gts']['pose'])):
                
                pred_motion = to_motion(pred_, mean, std, dim_pose)
                gt_motion = to_motion(gt_, mean, std, dim_pose)
                
                pred_rots = data_['preds']['pose_repr'][i]
                gt_rots = data_['gts']['pose_repr'][i]
                
                pred_motion_batch += [pred_motion]
                gt_motion_batch += [gt_motion]
                
                pred_motion = torch.cat((pred_motion, pred_motion[-1:]), dim=0)
                gt_motion = torch.cat((gt_motion, gt_motion[-1:]), dim=0)
                
                pred_motion_batch_enc += [pred_motion]
                gt_motion_batch_enc += [gt_motion]
                    
                pred_batch += [pred_]
                gt_batch += [gt_]
                
                #################
                pred_motion_emb_tmp = movement_enc(pred_motion.unsqueeze(0).cuda())[0].detach().cpu().numpy()
                gt_motion_emb_tmp = movement_enc(gt_motion.unsqueeze(0).cuda())[0].detach().cpu().numpy()
                
                # ----- root ------
                pred_wholebody_root_tmp = pred_[:, :1].copy()
                gt_wholebody_root_tmp = gt_[:, :1].copy()
                wholebody_root_l2_tmp = np.sqrt(((pred_wholebody_root_tmp - gt_wholebody_root_tmp) ** 2).sum(-1).astype(np.float32)).mean()
                data[batch_i]['preds']['root'][i] = wholebody_root_l2_tmp.item()
                
                # ----- mpjpe ------
                pred_wholebody_pose_centered_tmp = pred_.copy() - pred_wholebody_root_tmp
                gt_wholebody_pose_centered_tmp = gt_.copy() - gt_wholebody_root_tmp
                wholebody_mpjpe_tmp = np.sqrt(((pred_wholebody_pose_centered_tmp - gt_wholebody_pose_centered_tmp) ** 2).sum(-1).astype(np.float32)).mean()
                data[batch_i]['preds']['mpjpe'][i] = wholebody_mpjpe_tmp.item()
                
                # ----- fid on embeddings ----- 
                fid_tmp = [FIDCalculator.frechet_distance(pred_motion_emb_tmp, gt_motion_emb_tmp)]
                data[batch_i]['preds']['fid'][i] = np.array(fid_tmp).mean().item()
                
                # ----- angle mae ----- 
                angle_mae_tmp = get_ang_diff(pred_rots, gt_rots).detach().cpu().numpy()
                data[batch_i]['preds']['head_mae'][i] = np.array(angle_mae_tmp).mean().item()
                angle_mae_all += [angle_mae_tmp]
                
                # ---- score
                # loop over all participants
                chunk_id = int(data_['raw_dialogue'][i]['chunk_id'])
                obs_pose, tgt_pose = get_pose_from_raw_data(chunk_id, '4', data_raw, 5)
                for sub_i, obs_pose_ in enumerate(obs_pose):
            
                    obs_pose_rot = obs_pose_.copy()
                    
                    head_pos_obs = body_skeleton(torch.from_numpy(obs_pose_rot.copy()).cuda())
                    head_pos_obs = head_pos_obs[ -25:, DND_JOINT_NAMES.index('Head')].mean(dim=0).unsqueeze(0)
                    
                    gt_score = get_att_score(gt_rots, tgt_pose, head_pos_obs)
                    pred_score = get_att_score(pred_rots, tgt_pose, head_pos_obs)
                    
                    diff_score_all[sub_i] += [torch.abs(gt_score - pred_score).mean().detach().cpu().numpy()]
                    data[batch_i]['preds']['att_score_diff'][i][sub_i] = diff_score_all[sub_i][-1].item()
                #################
                
            if len(pred_motion_batch) != 0:    
                pred_motion_batch = torch.stack(pred_motion_batch, dim=0) # B, T', K
                gt_motion_batch = torch.stack(gt_motion_batch, dim=0) 
                
                pred_motion_all += [pred_motion_batch]
                gt_motion_all += [gt_motion_batch]
               
            if len(pred_motion_batch_enc) != 0:     
                pred_motion_batch_enc = torch.stack(pred_motion_batch_enc, dim=0) # B, T', K
                gt_motion_batch_enc = torch.stack(gt_motion_batch_enc, dim=0) 
                
                pred_motion_emb = movement_enc(pred_motion_batch_enc.cuda()).detach().cpu().numpy()
                gt_motion_emb = movement_enc(gt_motion_batch_enc.cuda()).detach().cpu().numpy()
                
                pred_motion_emb_all += [pred_motion_emb]
                gt_motion_emb_all += [gt_motion_emb]
               
            if len(pred_batch) != 0:
                pred_batch = np.stack(pred_batch, axis=0)
                gt_batch = np.stack(gt_batch, axis=0) 
                
                pred_all += [pred_batch]
                gt_all += [gt_batch]
              
        torch.save(data, file)
        
        pred_all = np.concatenate(pred_all, axis=0)
        gt_all = np.concatenate(gt_all, axis=0)
        
        pred_motion_all = np.concatenate(pred_motion_all, axis=0)
        gt_motion_all = np.concatenate(gt_motion_all, axis=0)
        
        gt_motion_emb_all = np.concatenate(gt_motion_emb_all, axis=0)
        pred_motion_emb_all = np.concatenate(pred_motion_emb_all, axis=0)
        
        # ====================
        # ----- root ------
        pred_wholebody_root = pred_all[:, :, :1].copy()
        gt_wholebody_root = gt_all[:, :, :1].copy()
        wholebody_root_l2 = np.sqrt(((pred_wholebody_root - gt_wholebody_root) ** 2).sum(-1).astype(np.float32)).mean()
        
        # ----- mpjpe ------
        pred_wholebody_pose_centered = pred_all.copy() - pred_wholebody_root
        gt_wholebody_pose_centered = gt_all.copy() - gt_wholebody_root
        wholebody_mpjpe = np.sqrt(((pred_wholebody_pose_centered - gt_wholebody_pose_centered) ** 2).sum(-1).astype(np.float32)).mean()
        
        # ----- diversity on motion chunks ----- 
        gt_diversity = calculate_avg_distance(gt_motion_all)
        pred_diversity = calculate_avg_distance(pred_motion_all)
        
        # ----- fid on embeddings ----- 
        fid = []
        for gt_motion_emb_, pred_motion_emb_ in zip(gt_motion_emb_all, pred_motion_emb_all):
            fid += [FIDCalculator.frechet_distance(pred_motion_emb_, gt_motion_emb_)]
        fid = np.array(fid).mean()
        
        # ----- angle mae ----- 
        angle_mae = np.concatenate(angle_mae_all).mean()
        
        # --- attention score difference
        for sub_i in range(4):
            diff_score_all[sub_i] = np.array(diff_score_all[sub_i]).mean()   

        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        with open(f'{dirname}/{set_name}_pose_metrics_{iter}.txt', 'w') as f:
            json.dump({'gt_diversity': gt_diversity.item(), \
                        'pred_diversity':pred_diversity.item(), \
                        'fid': fid.item(),\
                        'root_l2': wholebody_root_l2.item(),\
                        'mpjpe': wholebody_mpjpe.item(),\
                        'angle_mae': angle_mae.item(),\
                        'diff_att_socre_user1': diff_score_all[0].item(),\
                        'diff_att_socre_user2': diff_score_all[1].item(),\
                        'diff_att_socre_user3': diff_score_all[2].item(),\
                        'diff_att_socre_user4': diff_score_all[3].item(),\
                        }, f, indent=2)
    
    del movement_enc
 
from transformers import AutoProcessor, WavLMModel
from utils.styletts_syn import inference, load_styletts2_model
import torchaudio
def synthesize_speech_and_evaluate(tgt_dir, tokenizer, styletts_path, iter):
    
    processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    wavlm.cuda()
    wavlm.eval()
    
    alignmenter = alignment(sigma=1.25, order=12) # adjust sigma and order for ConvoFusion/MLD according to DiffGesture's jittery output
    
    device = f'cuda:0'
    model_params, styletts_model, sampler, global_phonemizer, textclenaer = load_styletts2_model(styletts_path, device)
    
    files = glob.glob(f'{tgt_dir}/*_results_{iter}.npy')

    for file in files:
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        if os.path.exists(f'{dirname}/{set_name}_speech_metrics_{iter}.txt') is True:
            continue
        
        data = torch.load(file, map_location="cpu", weights_only=False)
        
        sim_all = []
        gt_align_all = []
        pred_align_all = []
        for batch_i, data_ in enumerate(tqdm(data)):
            
            batch_size = len(data_['gts']['text_tok'])
            data[batch_i]['preds']['sim'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['align'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['align_diff'] = [[] for i in range(batch_size)]
            data[batch_i]['gts']['align'] = [[] for i in range(batch_size)]
            
            data[batch_i]['preds']['wav_true'] = [[] for i in range(batch_size)]
            data[batch_i]['preds']['wav'] = [[] for i in range(batch_size)]
            data[batch_i]['gts']['wav'] = [[] for i in range(batch_size)]
            
            pred_wav_recon_all = []
            gt_wav_recon_all = []
            
            for i, gt_text, in enumerate(data_['gts']['text_tok']):
                is_verbal = data_['raw_dialogue'][i]['conv_type'] in ['h1c1', 'h0c1']
                if is_verbal is False:
                    continue
                
                pred_text = data_['preds']['text_tok'][i]
                
                gt_style = data_['gts']['audio_emb'][i]
                
                # skip if it's nonverbal
                if len(gt_style) == 0:
                    continue
                
                pred_style = data_['preds']['audio_emb'][i]
                
                if len(pred_text) == 0:
                    pred_text = ""
                else:
                    pred_text = tokenizer.decode(pred_text.astype(np.int32), skip_special_tokens=True)
                
                gt_text = tokenizer.decode(gt_text.astype(np.int32), skip_special_tokens=True)
                
                # for style evaluation, we use gt text to ddcouple the two factors
                # texts are evaluated by WER
                pred_wav_out = inference(gt_text, torch.from_numpy(pred_style).to(device), alpha=0.1, beta=0.07, diffusion_steps=15, embedding_scale=1, sampler=sampler, global_phonemizer=global_phonemizer, textclenaer=textclenaer, model=styletts_model, device=device, model_params=model_params)
                gt_wav_out = inference(gt_text, torch.from_numpy(gt_style).to(device), alpha=0.1, beta=0.07, diffusion_steps=15, embedding_scale=1, sampler=sampler, global_phonemizer=global_phonemizer, textclenaer=textclenaer, model=styletts_model, device=device, model_params=model_params)
                
                pred_wav_out = torchaudio.functional.resample(pred_wav_out.unsqueeze(0).detach().cpu(), 24000, 16000)
                gt_wav_out = torchaudio.functional.resample(gt_wav_out.unsqueeze(0).detach().cpu(), 24000, 16000)
                
                data[batch_i]['preds']['wav'][i] = pred_wav_out.numpy().copy()
                data[batch_i]['gts']['wav'][i] = gt_wav_out.numpy().copy()
                
                pred_wav_recon_all += [pred_wav_out[0].numpy()]
                gt_wav_recon_all += [gt_wav_out[0].numpy()]
                
                ######################
                pred_wav_recon_all_tmp = [pred_wav_out[0].numpy()]
                gt_wav_recon_all_tmp = [gt_wav_out[0].numpy()]
                
                audio_tmp = pred_wav_recon_all_tmp + gt_wav_recon_all_tmp
                inputs_tmp = processor(audio_tmp, sampling_rate=16000, padding=True, return_tensors="pt")
                for k, v in inputs_tmp.items():
                    inputs_tmp[k] = v.cuda()
                
                sample_size = len(pred_wav_recon_all_tmp)
                embeddings_tmp = wavlm(**inputs_tmp)['extract_features']  
                embeddings_tmp = torch.nn.functional.normalize(embeddings_tmp, dim=-1)
                
                sim_tmp = torch.nn.functional.cosine_similarity(embeddings_tmp[:sample_size], embeddings_tmp[sample_size:], dim=-1).mean(dim=-1)
                data[batch_i]['preds']['sim'][i] = sim_tmp.detach().cpu().numpy().mean().item()
                
                ######################
                # --------- get audio pose beats and calculate alignment
                if len(data_['gts']['pose']) != 0:
                    
                    true_pred_wav_out = inference(pred_text, torch.from_numpy(pred_style).to(device), alpha=0.1, beta=0.07, diffusion_steps=15, embedding_scale=1, sampler=sampler, global_phonemizer=global_phonemizer, textclenaer=textclenaer, model=styletts_model, device=device, model_params=model_params)
                    
                    if true_pred_wav_out is not None:
                        true_pred_wav_out = torchaudio.functional.resample(true_pred_wav_out.unsqueeze(0).detach().cpu(), 24000, 16000)
                        data[batch_i]['preds']['wav_true'][i] = true_pred_wav_out.numpy().copy()
                        
                        pred_align = data_['preds']['pose'][i]
                        onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(true_pred_wav_out[0].numpy(), -1, -1, True)
                        
                        gt_align = data_['gts']['pose'][i]
                        onset_raw_gt, onset_bt_gt, onset_bt_rms_gt = alignmenter.load_audio(gt_wav_out[0].numpy(), -1, -1, True)
                            
                        if onset_raw is None or onset_raw_gt is None:
                            continue
                        else:
                            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = alignmenter.load_pose(pred_align, -1, -1, -1, True)
                            pred_align_all += [alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, 16000, 25)]
                            pred_align_all_tmp = pred_align_all[-1]
                            data[batch_i]['preds']['align'][i] = np.array(pred_align_all_tmp).mean().item() 
                            
                            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = alignmenter.load_pose(gt_align, -1, -1, -1, True)
                            gt_align_all += [alignmenter.calculate_align(onset_raw_gt, onset_bt_gt, onset_bt_rms_gt, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, 16000, 25)]
                            
                            gt_align_all_tmp = gt_align_all[-1]
                            data[batch_i]['gts']['align'][i] = np.array(gt_align_all_tmp).mean().item() 
                            
                            data[batch_i]['preds']['align_diff'][i] = (np.array(pred_align_all_tmp).mean() - np.array(gt_align_all_tmp).mean()).item() 
                            
                    else:
                        continue
            
            if len(gt_wav_recon_all) != 0:
                audio = pred_wav_recon_all + gt_wav_recon_all
                inputs = processor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
                
                sample_size = len(pred_wav_recon_all)
                embeddings = wavlm(**inputs)['extract_features']
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                
                sim = torch.nn.functional.cosine_similarity(embeddings[:sample_size], embeddings[sample_size:], dim=-1).mean(dim=-1)
                
                sim_all += [sim.detach().cpu().numpy()]
        
        torch.save(data, file)
        
        gt_align_final = np.array(gt_align_all).mean().item() if len(gt_align_all) != 0 else 0.0
        pred_align_final = np.array(pred_align_all).mean().item() if len(pred_align_all) != 0 else 0.0
        diff_align = np.abs(gt_align_final - pred_align_final).item()
        
        SIM_mean = np.concatenate(sim_all, axis=-1).mean() if len(sim_all) != 0 else 0.0
        
        dirname = os.path.dirname(file)
        set_name = str.split(str.split(file, '/')[-1], '_')[0]
        
        with open(f'{dirname}/{set_name}_speech_metrics_{iter}.txt', 'w') as f:
            json.dump({'speaker SIM': SIM_mean.item(), \
                        'gt_align': gt_align_final,\
                        'pred_align': pred_align_final,\
                        'abs_diff_align': diff_align,\
                        }, f, indent=2)  