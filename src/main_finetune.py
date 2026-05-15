import argparse
import datetime
import json
import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_finetune import train_one_epoch, evaluation, validation

from data.data_utils import MyDataCollate
from utils.dnd_skeleton import *
from data.dnd_config import *

import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
from model.polyslgen import PolySLGenWrapper
from torch.utils.data import DataLoader
from data.dnd_dataset import DnDDataset, FinetuneDistSampler, preprocess_data

from peft import get_peft_model
from finetunings.utils.config_utils import generate_peft_config
from utils.eval_metrics import calculate_eval_metrics 
               
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
    
    # skip if the training is finished
    if args.eval_only == 0 and len(glob.glob(os.path.join(args.output_dir, 'final_model', f'model_epoch_*.pth',))) == 0:
        
        # ========================
        # Initialization
        # ========================
        
        train_logger = misc.get_logger(args.output_dir, 'train_log')
        args.gpu = 0
        local_rank = args.gpu
        
        train_logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        train_logger.info("{}".format(args).replace(', ', ',\n'))
        
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        # ========================
        # Prepare data
        # ========================
        
        chunk_train_data = preprocess_data(args, tokenizer, partition='train', logger=train_logger)
        dataset_train = DnDDataset(args, chunk_train_data, tokenizer, partition = 'train')
        
        train_logger.info(f'[Train] dataset_train size: {len(dataset_train)}')
        
        sampler_train = FinetuneDistSampler(
            dataset_train, num_replicas=1, rank=0, shuffle=True, batch_size=args.batch_size,
            acc_grad=args.accum_iter
        )
        
        mydatacollate = MyDataCollate(args=args)
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            sampler=sampler_train,
            shuffle= (sampler_train is None),
            drop_last=True,
            collate_fn=mydatacollate.train_collate_fn,
        )
        train_logger.info(f'[Train] data_loader_train size: {len(data_loader_train)}')
        
        chunk_val_data = preprocess_data(args, tokenizer, partition='val', logger=train_logger)
        dataset_val = DnDDataset(args, chunk_val_data, tokenizer, partition = 'val')
        train_logger.info(f'[Train] dataset_val size: {len(dataset_val)}')
        
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            sampler = None,
            shuffle=False,
            drop_last=True,
            collate_fn=mydatacollate.train_collate_fn,
        )
        train_logger.info(f'[Train] data_loader_val size: {len(data_loader_val)}')
        
        # ========================
        # Load llama checkpoint
        # ========================
        train_logger.info(f'Loading model from {args.checkpoint_dir}/{args.llama_ckpt_dir}')
        
        model = AutoModelForCausalLM.from_pretrained(f'{args.checkpoint_dir}/{args.llama_ckpt_dir}',
                                                        quantization_config=None,
                                                        device_map="cpu",
                                                        attn_implementation= "eager",
                                                        torch_dtype=torch.bfloat16,
                                                        use_cache=False, 
                                                        pad_token_id = tokenizer.pad_token_id)
        for name, param in model.get_input_embeddings().named_parameters():
            param.data[tokenizer.pad_token_id] = torch.zeros_like(param[:1])
           
        #register_attention_control(model)   
        train_logger.info('Finished llama initialization!!')
        
        # If there is a mismatch between tokenizer vocab size and embedding matrix, 
        # throw a warning and then expand the embedding matrix
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            train_logger.info("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            model.resize_token_embeddings(len(tokenizer))
        
        # ----------------
        # Peft LoRA
        # ----------------
        params_to_update = {'r': args.lora_r, 
                               'lora_alpha': args.lora_alpha, 
                               'target_modules': args.lora_target_modules,
                               'bias': 'none',
                               'lora_dropout': args.lora_dropout,
                               'use_rslora': True,
                               'init_lora_weights': 'gaussian',
                               'task_type': "CAUSAL_LM",
                               'inference_mode': False}
     
        peft_config = generate_peft_config('lora', params_to_update)
        model = get_peft_model(model, peft_config)
        
        # init model
        model.print_trainable_parameters()
        # define the model
        model = PolySLGenWrapper(args, model, tokenizer, logger=train_logger)
        train_logger.info("Model = %s" % str(model))
            
        model.to(device=f'cuda:{local_rank}')
            
        eff_batch_size = args.batch_size * args.accum_iter
        train_logger.info("========== Effective batch size: %d =========='" % eff_batch_size)
        
        # ----------------
        # Optimizer
        # ----------------
        param_groups = {
            "decay": {"params": [], "weight_decay": args.weight_decay, "lr_scale": 0.5},
            "no_decay": {"params": [], "weight_decay": 0., "lr_scale": 0.5},
            "scratch_decay": {"params": [], "weight_decay": args.weight_decay, "lr_scale": 1.0},
            "scratch_no_decay": {"params": [], "weight_decay": 0., "lr_scale": 1.0},
        }
        train_logger.info("Making parameter groups ...")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = name.endswith(".bias") or name.endswith("norm.weight")
            scratch = "polyslgen.llama" not in name
            group_name = ("scratch_" if scratch else "") + ("no_decay" if no_decay else "decay")
            train_logger.info(f"{name}: in group {group_name}")
            param_groups[group_name]["params"].append(param)
        
        optimizer = optim.AdamW(
            [param_groups[key] for key in ["decay", "no_decay", "scratch_decay", "scratch_no_decay"]],
            betas=(0.9, 0.95), 
            lr = args.lr,
        )
        train_logger.info(optimizer)
        loss_scaler = NativeScaler(args)

        start_epoch = 0
        start_iter = 0
        if args.resume or args.auto_resume:
            start_epoch, start_iter = misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
        
        train_logger.info(f"Start training for {args.epochs} epochs")
        
        extra_epoch = start_iter // len(data_loader_train)
        start_iter = start_iter % len(data_loader_train)
        start_epoch = start_epoch + extra_epoch
        
        start_time = time.time()
        for epoch in range(start_epoch, args.epochs):
            
            data_loader_train.sampler.set_epoch(epoch, start_iter)
                
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, epoch, start_iter, loss_scaler, local_rank, 
                log_writer=log_writer,
                args=args,
                train_logger=train_logger,
            )
                
            train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            
            # validation
            if (epoch % args.val_interval == 0) or (epoch + 1 == args.epochs):
                val_stats = validation(
                    model, data_loader_val,
                    epoch, local_rank, 
                    log_writer=log_writer,
                    args=args,
                    mode='val',
                    train_logger=train_logger
                )
                val_log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch}
                    
            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(train_log_stats) + "\n")
                    if (epoch % args.val_interval == 0) or (epoch + 1 == args.epochs):
                        f.write(json.dumps(val_log_stats) + "\n")

            start_iter = 0 
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        train_logger.info('Training time {}'.format(total_time_str))
        
        # ====================
        # Save the final model
        # ====================
        
        model.polyslgen.llama.save_pretrained(f'{args.output_dir}/final_model/')
        del model.polyslgen.llama
        
        cpu_state = model.polyslgen.state_dict()
        train_logger.info(f"--> saving model ...")
        # create save path
        _epoch = args.epochs-1
        model_save_path = os.path.join(args.output_dir, 'final_model',f'model_epoch_{_epoch}.pth')
        # save model
        torch.save(cpu_state, model_save_path)
            
        train_logger.info(f"model checkpoint saved at {model_save_path}\n")   
        
        # remove previous ckpts
        ckpts = glob.glob(os.path.join(args.output_dir, "iter_*")) + glob.glob(os.path.join(args.output_dir, "epoch_*"))
        for ckpt in ckpts:
            train_logger.info(f'del {ckpt}')
            os.system(f'rm {ckpt} -rf')
                
        for handler in train_logger.handlers:
            train_logger.removeHandler(handler)
            handler.close()
            
        del model
        del optimizer

        # model will still be on cache until its place is taken by other objects so also execute the below lines
        gc.collect()
        torch.cuda.empty_cache() 
    
    # run testing
    eval_main(args, tokenizer)
    
    # calculate metrics for each iteration
    calculate_eval_metrics(args, tokenizer)
                 
    
def eval_main(args, tokenizer):
    
    test_logger = misc.get_logger(args.output_dir, 'test_log')
     
    args.gpu = 0
    torch.cuda.set_device(0)
    misc.set_random_seeds(42)
        
    test_logger.info("Loading pretrained weights ...")
    
    from peft import PeftConfig, PeftModel
    
    if args.eval_only != 0:
        pretrained_path = glob.glob(f'{args.pretrained_model_dir}/final_model/model_epoch_*.pth')[0]
    else:
        pretrained_path = glob.glob(f'{args.output_dir}/final_model/model_epoch_*.pth')[0]
    test_logger.info("Resume checkpoint %s" % pretrained_path)
    
    config = PeftConfig.from_pretrained(os.path.dirname(pretrained_path))
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, os.path.dirname(pretrained_path))
    
    model = PolySLGenWrapper(args, model, tokenizer, logger=test_logger)
    
    checkpoint_llama_wrapper = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    msg = model.polyslgen.load_state_dict(checkpoint_llama_wrapper, strict=False)
    test_logger.info(f"load result:\n{msg}")
        
    model.to(device='cuda')
    model.eval()
    test_logger.info(f"Model = {str(model)}")
    
    for ii in range(args.test_iter):
        
        if os.path.exists(os.path.join(args.output_dir, f"test_results_{ii}.npy")) is True:
            continue
            
        chunk_test_data = preprocess_data(args, model.polyslgen.tokenizer, partition='test', logger=test_logger)
        mydatacollate = MyDataCollate(args=args)
        dataset_test = DnDDataset(args, chunk_test_data, tokenizer, partition = 'test')
        test_logger.info(f'[Test] dataset_test size: {len(dataset_test)}')
        
        data_loader_test = DataLoader(
            dataset_test,
            batch_size= args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            shuffle=False,
            drop_last=False,
            collate_fn=mydatacollate.test_collate_fn,
        )
        test_logger.info(f'[Test] data_loader_test size: {len(data_loader_test)}')
        
        # evaluation
        start_test_time = time.time()
        test_stats, final_outputs = evaluation(
            model, data_loader_test,
            args.epochs,
            args=args,
            mode='test',
            test_logger=test_logger
            )
        
        total_test_time = time.time() - start_test_time
        total_test_time_str = str(datetime.timedelta(seconds=int(total_test_time)))
        test_logger.info('Testing time {}'.format(total_test_time_str))

        test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': args.epochs, 'time': total_test_time_str}
        
        if args.output_dir:
            with open(os.path.join(args.output_dir, f"test_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(test_log_stats) + "\n") 
            torch.save(final_outputs, os.path.join(args.output_dir, f"test_results_{ii}.npy"))
                
    for handler in test_logger.handlers:
        test_logger.removeHandler(handler)
        handler.close()
    
          
if __name__ == '__main__':
    
    misc.set_random_seeds(42)
    
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)