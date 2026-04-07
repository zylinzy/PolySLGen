import os


src_dir = 'path/to/the/main/code/folder/src/'

exp_name = 'exp_polyslgen_full'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#os.environ["PATH"] += ":path/to/espeak/usr/bin" # specify espeak path if espeak is installed somewhere else
#os.environ["ESPEAK_DATA_PATH"] = "path/to/espeak/usr/share/espeak-ng-data" # specify espeak path if espeak is installed somewhere else
#os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "path/to/espeak/usr/lib64/libespeak-ng.so.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_code = f'{src_dir}/main_finetune.py'
#train_code = f'{src_dir}/main_baselines.py' # run NN and RANDOM baselines

### In/Out settings ###
data_dir = f'{src_dir}/../dnd_dataset_preprocessed/' # dataset directory
output_dir = f'{src_dir}/../results/{exp_name}'
log_dir = output_dir

### Checkpoints ###
llama_ckpt_dir = 'Meta-Llama-3-8B-Instruct' # directory name of the llama model
checkpoint_dir = f'{src_dir}/../checkpoints/'

motion_evaluator_path = f'{checkpoint_dir}/motion_evaluator/model/motion_evaluator_c64_E0150.tar'
styletts_path = f'{checkpoint_dir}/StyleTTS2/'
dnd_joint_init_path = f'{src_dir}/utils/dnd_joint_skeleton.npy'

### Data settings ###
chunk_length = 64
pose_hist_length = 64
hist_length = 512

### Training settings ###
epochs = 20
batch_size = 4
accum_iter = 4
max_words = 1024
lr = 0.0002

### LoRA settings ###
lora_target_modules = 'q_proj v_proj k_proj'
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

### Pose fusion ###     
pose_fusion = 1

### Social cue encoder ###
social_cue = 1

### Evaluation ###
test_batch_size = 16
test_iter = 5  # number of iterations for evaluation
eval_only = 0  # set to 1 if you only want to run evaluation on a given pre-trained model
pretrained_model_dir = f'{src_dir}/../pretrained_model/' # path to the folder for pre-trained model

cmd = f'python3.9 {train_code}\
            --pin_mem \
            --epochs {epochs} \
            --batch_size {batch_size} \
            --data_dir {data_dir}\
            --output_dir {output_dir} \
            --log_dir {log_dir}\
            --llama_ckpt_dir {llama_ckpt_dir}\
            --checkpoint_dir {checkpoint_dir}\
            --num_workers 4\
            --chunk_length {chunk_length}\
            --hist_length {hist_length}\
            --max_words {max_words} \
            --warmup_epochs 0.05 \
            --accum_iter {accum_iter}\
            --auto_resume \
            --weight_decay 0.0 \
            --lr {lr} \
            --min_lr 0.0 \
            --lora_target_modules {lora_target_modules}\
            --lora_r {lora_r}\
            --lora_alpha {lora_alpha}\
            --lora_dropout {lora_dropout}\
            --test_batch_size {test_batch_size}\
            --motion_evaluator_path {motion_evaluator_path}\
            --styletts_path {styletts_path}\
            --pose_fusion {pose_fusion}\
            --pose_hist_length {pose_hist_length}\
            --social_cue {social_cue}\
            --test_iter {test_iter}\
            --dnd_joint_init_path {dnd_joint_init_path}\
            --eval_only {eval_only}\
            --pretrained_model_dir {pretrained_model_dir}\
            '

os.system(cmd)
