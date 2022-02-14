import random
import torch
import json
import numpy as np
from tqdm import tqdm

def load_data(file_path, mode="train"):
    same_cnt = 0
    src_lst, trg_lst = [], []

    with open(file_path, "r", encoding="utf8") as fr:
        pairs = json.load(fr)
        for pair in pairs:
            err_sent = pair["original_text"]
            cor_sent = pair["correct_text"]

            assert len(err_sent) == len(cor_sent)  #check if trg length == src length
            if err_sent == cor_sent:
                same_cnt += 1

            src_lst.append(err_sent)
            trg_lst.append(cor_sent)

    print(f"{len(src_lst)} examples in {mode} file, {same_cnt} pairs are the same")
    return src_lst, trg_lst

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def write_to_file(file_path, results):
    result_j = []
    for result in results:
        result_j.append({
            "err_sent": " ".join(result[0]),
            "cor_sent": " ".join(result[1]),
            "pre_sent": " ".join(result[2])
        })
    json.dump(result_j, open(file_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)

def compute_model_size(model):
    total_params, trainable_params, non_trainable_params = 0, 0, 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        total_params += mulValue
        if param.requires_grad:
            trainable_params += mulValue
        else:
            non_trainable_params += mulValue
    print(model)
    print(f"Total params:  {total_params}, Trainable params:  {trainable_params}, "
          f"Nontrainable params:  {non_trainable_params}.")