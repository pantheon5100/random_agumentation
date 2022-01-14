# zero + norm
python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 32 \
        --alpha 32 \
        --delta-init zero \
        --out-dir "FGSM_eps32/StepSize32_zero_imgNorm" \
        --seed 0 \
        --image_normalize

# zero + non-norm
python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 32 \
        --alpha 32 \
        --delta-init zero \
        --out-dir "FGSM_eps32/StepSize32_zero_RemoveNorm" \
        --seed 0 

# random + norm
python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 32 \
        --alpha 32 \
        --delta-init random \
        --out-dir "FGSM_eps32/StepSize32_random_imgNorm" \
        --seed 0 \
        --image_normalize

# random + non-norm
python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 32 \
        --alpha 32 \
        --delta-init random \
        --out-dir "FGSM_eps32/StepSize32_random_RemoveNorm" \
        --seed 0 