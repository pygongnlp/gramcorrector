CUDA_VISIBLE_DEVICES=0 nohup python ged_classification/train.py \
          --model_name_or_path bert-base-chinese \
          --train_batch_size 128 \
          --valid_batch_size 64 \
          --epoch 100 \
          --lr 3e-5 \
          --patience 3 \
          --output_dir ged_classification/checkpoints/bert \
          > log/sged_bert.log 2>&1 &