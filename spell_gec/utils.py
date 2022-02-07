import random
import torch
import json
import numpy as np

def load_data(file_path, mode="train"):
    same_cnt = 0
    src_lst, trg_lst = [], []

    with open(file_path, "r", encoding="utf8") as fr:
        for pair in fr.readlines():
            pair = pair.strip("\n")
            pair = pair.split()
            assert len(pair) == 2

            err_sent, cor_sent = pair

            assert len(err_sent) == len(cor_sent)  #check if trg length == src length
            if err_sent == cor_sent:
                same_cnt += 1

            src_lst.append(err_sent)
            trg_lst.append(cor_sent)

    print(f"{len(src_lst)} examples in {mode} file, {same_cnt} pairs are the same")
    return src_lst, trg_lst

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def write_to_file(file_path, results):
    def convert_to_json(results):
        result_j = []
        for result in results:
            result_j.append({
                "error": result[0],
                "correct": result[1],
                "predict": result[2]
            })
        return result_j
    json.dump(convert_to_json(results), open(file_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)

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