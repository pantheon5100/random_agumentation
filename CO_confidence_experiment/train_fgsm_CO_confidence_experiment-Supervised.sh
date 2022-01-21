
python train_fgsm_CO_confidence_experiment.py --batch-size 128 \
                --data-dir /dev/shm \
                --epochs 30 \
                --lr-schedule cyclic \
                --lr-min 0. \
                --lr-max 0.3 \
                --weight-decay 5e-4 \
                --epsilon 16 \
                --alpha 20 \
                --delta-init zero \
                --out-dir "train_fgsm_CO_confidence_experiment-Supervised" \
                --seed 0 \
                --supervised \
                --image_normalize


