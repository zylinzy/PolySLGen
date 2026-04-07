# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import glob
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from torch import inf

from types import TracebackType
from typing import Any, Optional, Type
import torch

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, start_iter=0, logger=None):
        i = start_iter
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        log_msg = [
            header,
            '[{0' + '}/{1}]',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                try:
                    total_len = len(iterable)
                except:
                    total_len = "unknown"
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, total_len,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, total_len,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

class NativeScalerWithGradNormCount:
    
    def __init__(self, args):
        self.use_fp16 = False
        self.clip_value = 100000 if args.clip_grad < 0 else args.clip_grad
        self.is_clip = args.clip_grad >= 0

    def __call__(self, loss, optimizer, model, parameters=None, create_graph=False, update_grad=True):
        loss.backward()
        if update_grad:
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.clip_value)
            optimizer.step()
        else:
            norm = None
        
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(output_dir, args, epoch, iteration, model, optimizer, loss_scaler, dataset_state):
    save_dir = os.path.join(output_dir, f"epoch_{epoch:03d}_iter_{iteration:09d}")
    os.makedirs(save_dir, exist_ok=True)
    
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler.use_fp16 else None,
        "args": args,
    }
    save_path = os.path.join(
        save_dir,
        f"checkpoint.00000-of-00001.pth",
    )
    torch.save(to_save, save_path)
    
    # remove previous ckpts
    ckpts = glob.glob(os.path.join(output_dir, "iter_*")) + glob.glob(os.path.join(output_dir, "epoch_*"))
    ckpts.sort()
    if len(ckpts)>2:
        for ckpt in ckpts[:-2]:
            print('del', ckpt)
            os.system(f'rm {ckpt} -rf')

def load_model(args, model, optimizer, loss_scaler):
    start_iter = 0
    start_epoch = 0
    if args.auto_resume:
        ckpt_dirs = glob.glob(os.path.join(args.output_dir, "iter_*")) + glob.glob(os.path.join(args.output_dir, "epoch_*"))
        ckpt_dirs.sort()
        if len(ckpt_dirs) > 0:
            args.resume = ckpt_dirs[-1]
    if args.resume:
        print("Resume checkpoint %s" % args.resume)
        local_checkpoint_path = os.path.join(
            args.resume,
            f"checkpoint.{get_rank():05d}-of-{get_world_size():05d}.pth",
        )
        print('local_checkpoint_path:', local_checkpoint_path)
        checkpoint = torch.load(local_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
            
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = int(checkpoint['iter']) + 1
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch'])
    return start_epoch, start_iter
    
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().cuda()
        else:
            x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        #if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
        if name.endswith(".bias") or name.endswith("norm.weight"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]




class default_tensor_type:
    _tensor_type_stack = [(torch.float, "cpu")]
    
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        # Only limited combinations are supported.
        assert device is None or device in ["cpu", "cuda"]
        assert dtype is None or dtype in [torch.float, torch.bfloat16, torch.half]
        self.dtype, self.device = dtype, device
    
    def __enter__(self) -> None:
        dtype, device = self.dtype, self.device
        if dtype is None:
            dtype = default_tensor_type._tensor_type_stack[-1][0]
        if device is None:
            device = default_tensor_type._tensor_type_stack[-1][1]
        default_tensor_type._tensor_type_stack.append((dtype, device))
        
        # We use all 3 calls since the new apis (set_default_device, set_default_dtype)
        # seems to be ineffective sometimes (e.g., set_default_device is ineffective to
        # torch.Tensor calls).
        torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        default_tensor_type._tensor_type_stack.pop()
        dtype, device = default_tensor_type._tensor_type_stack[-1]

        torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)

    @staticmethod
    def get_tensor_type(dtype: torch.dtype, device: str) -> Any:
        return {
            (torch.float, "cpu"): torch.FloatTensor,
            (torch.bfloat16, "cpu"): torch.BFloat16Tensor,
            (torch.half, "cpu"): torch.HalfTensor,
            (torch.float, "cuda"): torch.cuda.FloatTensor,
            (torch.bfloat16, "cuda"): torch.cuda.BFloat16Tensor,
            (torch.half, "cuda"): torch.cuda.HalfTensor,
        }[(dtype, device)]


from transformers.utils.import_utils import *
# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py
import numpy as np
import random

def set_random_seeds(random_seed):
    
    os.environ['PYTHONHASHSEED']=str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if is_torch_mlu_available():
        torch.mlu.manual_seed_all(random_seed)
    if is_torch_musa_available():
        torch.musa.manual_seed_all(random_seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(random_seed)
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(random_seed)
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(random_seed)
        tf.config.experimental.enable_op_determinism()
            
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # The environment variable required to enable deterministic mode on Ascend NPUs.
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["HCCL_DETERMINISTIC"] = "1"

    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["PL_GLOBAL_SEED"] = str(random_seed)
    os.environ["PL_SEED_WORKERS"] = f"1"
    
    
import logging

def get_logger(out_dir, filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, f"{filename}.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    return logger