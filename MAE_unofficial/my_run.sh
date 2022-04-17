# Set the path to save checkpoints
# OUTPUT_DIR='./results/cifar-10'
# DATA_PATH='/media/ubuntu204/F/Dataset/cifar-10'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_mae_pretraining.py \
#         --data_path ${DATA_PATH} \
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch8_32 \
#         --batch_size 64 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 10 \
#         --epochs 200 \
#         --output_dir ${OUTPUT_DIR} \
#         --log_dir ${OUTPUT_DIR}

clear
OUTPUT_DIR='./results/tmp/'$(date +%Y%m%d_%H%M%S)
if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir $OUTPUT_DIR
fi

# OUTPUT_DIR='./results/tmp/ILSVRC2012-100'
DATA_PATH='/media/ubuntu204/F/Dataset/ILSVRC2012-10/train'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 my_run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.1 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 16 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --epochs 200 \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}

# OUTPUT_DIR='./results/tmp/finetune'
# DATA_PATH='/media/ubuntu204/F/Dataset/ILSVRC2012-100'
# MODEL_PATH='./results/pretrained/pretrain_mae_vit_base_mask_0.75_400e.pth'
# # batch_size can be adjusted according to the graphics card
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_class_finetuning.py \
#     --model vit_base_patch16_224 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 128 \
#     --opt adamw \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 3 \
#     --dist_eval

# # Set the path to save images
# OUTPUT_DIR='./results/tmp/visual'
# # path to image for visualization
# IMAGE_PATH='/media/ubuntu204/F/Dataset/ILSVRC2012-100/val/n01440764/ILSVRC2012_val_00000293.JPEG'
# # path to pretrain model
# MODEL_PATH='./results/pretrained/pretrain_mae_vit_base_mask_0.75_400e.pth'

# # Now, it only supports pretrained models with normalized pixel targets
# python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}