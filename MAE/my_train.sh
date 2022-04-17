clear
OUTPUT_DIR='./results/train/'$(date +%Y%m%d_%H%M%S)
if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir $OUTPUT_DIR
fi

IMAGENET_DIR='/media/ubuntu204/F/Dataset/ILSVRC2012-2'
PRETRAIN_CHKPT='./models/mae_pretrain_vit_base.pth'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 my_main_pretrain.py \
#     --accum_iter 4 \
#     --batch_size 32 \
#     --model vit_base_patch16 \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#     --dist_eval --data_path ${IMAGENET_DIR}

# python my_submitit_pretrain.py \
#     --job_dir ${JOB_DIR} \
#     --nodes 1 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --accum_iter 16 \
#     --data_path ${IMAGENET_DIR}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 my_main_pretrain.py \
    --batch_size 32 \
    --accum_iter 8 \
    --epochs 200 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \