CUDA_VISIBLE_DEVICES=0 python tged/test.py \
                --model_name_or_path bert-base-chinese \
                --save_path tged/checkpoints/bert \
                --test_file data/sighan/test.json