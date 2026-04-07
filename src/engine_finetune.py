import math
import sys

import torch
import torch.distributed as dist

import utils.misc as misc
import utils.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter, loss_scaler, local_rank,
                    log_writer=None,
                    args=None, train_logger=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    device = torch.device(f'cuda:{local_rank}')
    
    if log_writer is not None:
        train_logger.info('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, data  in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter, logger=train_logger), start=start_iter
    ):
        total_data_iter_step = epoch * len(data_loader) + data_iter_step
        
        _, input_tokens_pad, labels, target_mask_pad, modality_mask_pad, audio_data, pose_data, speaking_state = data
            
        input_tokens_pad = input_tokens_pad.to(device=device).long()
        labels = labels.to(device=device).long()
        target_mask_pad = target_mask_pad.to(device=device).long()
        modality_mask_pad = modality_mask_pad.to(device=device).long()
        speaking_state = speaking_state.to(device=device).float()
        
        for k, v in audio_data.items():
            if isinstance(audio_data[k], list):
                audio_data[k] = [ ele.to(device=device) for ele in audio_data[k]]
            else:
                audio_data[k] = audio_data[k].to(device=device)
                
        for k, v in pose_data.items():
            if isinstance(pose_data[k], list):
                pose_data[k] = [ ele.to(device=device) for ele in pose_data[k]]
            else:
                pose_data[k] = pose_data[k].to(device=device)
        
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or (data_iter_step == len(data_loader) - 1)
        if data_iter_step == 0 or update_grad == 0:
            lr_sched.adjust_learning_rate_epoch(optimizer, float(data_iter_step) / float(len(data_loader)) + epoch, args)
        
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
        
        with autocast_ctx:
            losses = model(input_tokens_pad, target_mask_pad, modality_mask_pad, audio_data, pose_data, labels=labels, speaking_state=speaking_state)
        
        loss_all = losses['loss_all']
        loss_all_value = losses['loss_all'].item()
        loss_text_value = losses['loss_text'].item()
        loss_audio_value = losses['loss_audio'].item()
        loss_body_value = losses['loss_body'].item()
        loss_body_root_l2_value = losses['loss_body_root_l2'].item()
        loss_body_mpjpe_value = losses['loss_body_mpjpe'].item()
        loss_body_reg_value = losses['loss_body_reg'].item()
        loss_state_value = losses['loss_speaking_state'].item()
                
        if not math.isfinite(loss_all_value):
            train_logger.info("[Rank {}] i_loss is {}, stopping training at {}".format(dist.get_rank(), loss_all_value, data_iter_step), force=True)
            for k, v in losses.items():
                train_logger.info(k, v.item())
            
            sys.exit(1)
        
        grad_norm = loss_scaler(
                loss_all / accum_iter, optimizer, model,
                parameters=model.parameters(),
                update_grad=update_grad,
            )
            
        if update_grad:
            assert grad_norm is not None
            metric_logger.update(grad_norm=grad_norm)
            optimizer.zero_grad()

        metric_logger.update(loss_all=loss_all_value)
        metric_logger.update(loss_text=loss_text_value)
        metric_logger.update(loss_audio=loss_audio_value)
        metric_logger.update(loss_body=loss_body_value)
        metric_logger.update(loss_body_root_l2=loss_body_root_l2_value)
        metric_logger.update(loss_body_mpjpe=loss_body_mpjpe_value)
        metric_logger.update(loss_body_reg=loss_body_reg_value)
        metric_logger.update(loss_state=loss_state_value)
            
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # save checkpoint
        is_last_epoch = (epoch + 1 == int(args.epochs))
        is_save_epoch = (epoch % int(args.save_interval) == 0)
        is_last_iter = (data_iter_step+1 == len(data_loader))
        
        save_epoch = (is_save_epoch or is_last_epoch) and is_last_iter
        save_intermediate  = (data_iter_step % 1000 == 0) and (data_iter_step != 0)
        if args.output_dir and (save_epoch or save_intermediate):
            misc.save_model(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None)
            
        if update_grad:
            loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
            loss_text_value_reduce = misc.all_reduce_mean(loss_text_value)
            loss_audio_value_reduce = misc.all_reduce_mean(loss_audio_value)
            loss_body_value_reduce = misc.all_reduce_mean(loss_body_value)
            loss_body_root_l2_value_reduce = misc.all_reduce_mean(loss_body_root_l2_value)
            loss_body_mpjpe_value_reduce = misc.all_reduce_mean(loss_body_mpjpe_value)
            if update_grad:
                grad_norm_reduce = misc.all_reduce_mean(grad_norm)
                
            loss_body_reg_value_reduce = misc.all_reduce_mean(loss_body_reg_value)
            loss_state_value_reduce = misc.all_reduce_mean(loss_state_value)
            

        if log_writer is not None and update_grad:
            log_writer.add_scalar('train/total_loss', loss_all_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_text', loss_text_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_audio', loss_audio_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_body', loss_body_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_body_root_l2', loss_body_root_l2_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_body_mpjpe', loss_body_mpjpe_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_body_reg', loss_body_reg_value_reduce, total_data_iter_step)
            log_writer.add_scalar('train/loss_state', loss_state_value_reduce, total_data_iter_step)
                
            if update_grad:
                log_writer.add_scalar('train/grad_norm', grad_norm_reduce.detach().float(), total_data_iter_step)
            log_writer.add_scalar('train/lr', lr, total_data_iter_step)
        
    train_logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validation(model, data_loader, epoch, local_rank, log_writer=None, args=None, mode='val', train_logger=None):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '***{}*** Epoch: [{}]'.format(mode, epoch)
    print_freq = 10

    device = torch.device(f'cuda:{local_rank}')
    
    if log_writer is not None:
        train_logger.info('log_dir: {}'.format(log_writer.log_dir))
        
    start_iter = 0
    current_total_iter = epoch * len(data_loader) + len(data_loader) - 1
    with torch.inference_mode():
        for _, data  in enumerate(
            metric_logger.log_every(data_loader, print_freq, header, start_iter, logger=train_logger), start=start_iter):
            
            _, input_tokens_pad, labels, target_mask_pad, modality_mask_pad, audio_data, pose_data, speaking_state = data
                
            input_tokens_pad = input_tokens_pad.to(device=device).long()
            labels = labels.to(device=device).long()
            target_mask_pad = target_mask_pad.to(device=device).long()
            modality_mask_pad = modality_mask_pad.to(device=device).long()
            speaking_state = speaking_state.to(device=device).float()
            
            for k, v in audio_data.items():
                if isinstance(audio_data[k], list):
                    audio_data[k] = [ ele.to(device=device) for ele in audio_data[k]]
                else:
                    audio_data[k] = audio_data[k].to(device=device)
                
            for k, v in pose_data.items():
                if isinstance(pose_data[k], list):
                    pose_data[k] = [ ele.to(device=device) for ele in pose_data[k]]
                else:
                    pose_data[k] = pose_data[k].to(device=device)
            
            autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
            
            with autocast_ctx:
                losses = model(input_tokens_pad, target_mask_pad, modality_mask_pad, audio_data, pose_data, labels=labels, is_train=False, speaking_state=speaking_state)
            
            loss_all_value = losses['loss_all'].item()
            loss_text_value = losses['loss_text'].item()
            loss_audio_value = losses['loss_audio'].item()
            loss_body_value = losses['loss_body'].item()
            loss_body_root_l2_value = losses['loss_body_root_l2'].item()
            loss_body_mpjpe_value = losses['loss_body_mpjpe'].item()
            loss_body_reg_value = losses['loss_body_reg'].item() 
            loss_state_value = losses['loss_speaking_state'].item()
            
            eval_body_root_l2_value = losses['eval_body_root_l2'].item()
            eval_body_mpjpe_value = losses['eval_body_mpjpe'].item()
            eval_text_tok_acc_value = losses['eval_text_tok_acc'].item()
        

            metric_logger.update(loss_all=loss_all_value)
            metric_logger.update(loss_text=loss_text_value)
            metric_logger.update(loss_audio=loss_audio_value)
            metric_logger.update(loss_body=loss_body_value)
            metric_logger.update(loss_body_root_l2=loss_body_root_l2_value)
            metric_logger.update(loss_body_mpjpe=loss_body_mpjpe_value)
            metric_logger.update(loss_body_reg=loss_body_reg_value)
            metric_logger.update(loss_state=loss_state_value)
            metric_logger.update(eval_body_root_l2=eval_body_root_l2_value)
            metric_logger.update(eval_body_mpjpe=eval_body_mpjpe_value)
            metric_logger.update(eval_text_tok_acc=eval_text_tok_acc_value)
        
    model.train()
    
    if log_writer is not None:
        for k, meter in metric_logger.meters.items():
            log_writer.add_scalar(f'val/{k}', meter.global_avg, current_total_iter)
            
    train_logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluation(model, data_loader, epoch, args=None, mode='test', test_logger=None):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '***{}*** Epoch: [{}]'.format(mode, epoch)
    print_freq = 1
    
    rank = next(model.parameters()).device
        
    start_iter = 0
    final_outputs = []
    with torch.inference_mode():
        for _, data  in enumerate(
            metric_logger.log_every(data_loader, print_freq, header, start_iter, logger=test_logger), start=start_iter
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
                
            
            preds, gts = model.generate_and_collect(input_tokens, target_mask, modality_mask, audio_data, pose_data, \
                                labels=labels, speaking_states=speaking_state)
            
            # save testing results
            final_outputs.append({'preds': preds, 'gts': gts, 'raw_dialogue': raw_dialogue})
            
            
    model.train()
    
    test_logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, final_outputs