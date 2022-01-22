
# save the checkpoint where CO happening and the final model

python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 20 \
        --delta-init zero \
        --out-dir "FGSM_baseline-eps16-step_size20" \
        --seed 0 \
        --image_normalize



python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 20 \
        --delta-init random \
        --out-dir "FGSM_RS-eps16-step_size20" \
        --seed 0 \
        --image_normalize

