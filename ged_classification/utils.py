import random
import torch
import json
import numpy as np

def load_data(file_path, mode="train"):
    same_cnt = 0
    sents, labels = [], []

    with open(file_path, "r", encoding="utf8") as fr:
        pairs = json.load(fr)
        for pair in pairs:
            err_sent = pair["original_text"]
            cor_sent = pair["correct_text"]
            if err_sent == cor_sent:
                labels.append("T")
                same_cnt += 1
            else:
                labels.append("F")
            sents.append(err_sent)

    print(f"{len(sents)} examples in {mode} file, {same_cnt} pairs are the same")
    return sents, labels

def write_to_file(results, file_path):
    result_lst = []
    for result in results:
        result_lst.append(
            {
                "sentence": result[0],
                "predict_label": result[1],
                "true_label": result[2]
            }
        )
    json.dump(result_lst, open(file_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
    print(f"write to {file_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_model_size(model):
    total_params, trainable_params, non_trainable_params = 0, 0, 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        total_params += mulValue
        if param.requires_grad:
            trainable_params += mulValue
        else:
            non_trainable_params += mulValue
    #print(model)
    print(f"Total params:  {total_params}, Trainable params:  {trainable_params}, "
          f"Nontrainable params:  {non_trainable_params}.")