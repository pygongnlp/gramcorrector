CUDA_VISIBLE_DEVICES=0 nohup python spell_gec/train.py \
          --train_batch_size 128 \
          --valid_batch_size 64 \
          --epoch 100 \
          --lr 3e-5 \
          --patience 3 \
          --output_dir spell_gec/checkpoints/bert \
          > log/spell_gec_bert.log 2>&1 &