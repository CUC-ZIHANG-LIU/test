export MODEL_NAME="/home/chenghaonan/qqt/AUDIT_my/checkpoint/model_checkpoints/pipeline"
export TRAIN_DIR=""
#CUDA_VISIBLE_DEVICES=1,2,3  accelerate launch  train_audit_e.py \
#CUDA_VISIBLE_DEVICES=3,4,5  accelerate launch  train_audit_e-va-r1-para.py \train_audit_e-va-r1-para-f0   #--mixed_precision="fp16"\  train_audit_e-va-r1-para-f0-freezproj.py train_audit_e-oir-f0.py
CUDA_VISIBLE_DEVICES=8,9  accelerate launch  train_audit_e-va-r1-para-f0.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=6 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --learning_rate=2e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/vta-lr2e5-10000-f0-constant-batch6"\ 
#train_audit_e-va-r1.py
# export MODEL_NAME="/home/chenghaonan/qqt/AUDIT_my/checkpoint/model_checkpoints/pipeline"
# export TRAIN_DIR=""
 #./data_check_results  --output_dir="/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/va-loss01"\
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch train_audit_e-va-r1.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --max_train_steps=100000 \
#   --checkpointing_steps=2000 \
#   --learning_rate=1e-5 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir "/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/va-loss03"#./data_check_results  --output_dir="/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/va-loss01"\



# CUDA_VISIBLE_DEVICES=1,2 accelerate launch train_audit_e-my-lorare.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --max_train_steps=1000000 \
#   --checkpointing_steps=20 \
#   --learning_rate=1e-5 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/oir-test"\

# accelerate launch train_audit_e-my-lorare-GPUre.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --use_ema \
#   --mixed_precision fp16 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --max_train_steps=1000000 \
#   --checkpointing_steps=20 \
#   --learning_rate=1e-5 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="/home/chenghaonan/qqt/AUDIT_my/checkpoint-lora/oir-test"\




#   export MODEL_NAME="/blob/v-yuancwang/AudioEditingModel/Diffusion_SG/checkpoint-10000"
# export TRAIN_DIR=""

# accelerate launch /home/v-yuancwang/AUDIT_v2/como/train_audit_cd.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=6 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --max_train_steps=1000000 \
#   --checkpointing_steps=2000 \
#   --learning_rate=2e-5 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="/blob/v-yuancwang/AUDITPLUS/AUDIT_CD_100" \