for i in 0 1
do
python train_fgsm_NoiseAug_later_random_start.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 8 \
        --alpha 8 \
        --delta-init zero \
        --out-dir later_random_start \
        --seed $i \
        --image_normalize
done

