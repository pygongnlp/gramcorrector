import os
import torch
import argparse
import numpy as np

from transformers import AutoConfig, AutoModel, AutoTokenizer
from model import GED
from utils import load_data

def check_args(args):
    assert os.path.exists(args.model_name_or_path)
    assert args.device in ["cpu", "cuda"]
    assert os.path.exists(args.model_path)
    assert args.interact or os.path.exists(args.test_file)
    #assert not args.interact and not os.path.exists(args.test_file)
    assert args.max_length < 512 and args.max_length > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="model/chinese_bert", type=str)
    parser.add_argument("--model_path", default="ged_classification/output/bert_ged.tar", type=str)
    parser.add_argument("--test_file", default="data/sighan/15/15test.txt", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--interact", action="store_true")
    args = parser.parse_args()
    check_args(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cpu":
        device = torch.device("cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
    else:
        checkpoint = torch.load(args.model_path)

    label_category = checkpoint["label_category"]
    num_labels = len(label_category)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pretrained_model = AutoModel.from_config(config)
    model = GED(
        pretrained_model=pretrained_model,
        hidden_size=config.hidden_size,
        num_labels=num_labels
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.device == "cuda":
        model.to(device)
    print("Load model state dict")

    model.eval()
    id2label = {0: "正确", 1: "错误"}
    if args.interact:
        sentence = input("输入句子:\t")
        input = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=args.max_length)
        logits = model(input).logits[0]
        logits_softmax = torch.softmax(logits, -1).tolist()
        print(f"这句话语法{id2label[torch.argmax(logits, -1).item()]}")
        print(f"正确概率{logits_softmax[0] * 100:4f}%, 错误概率{logits_softmax[1] * 100:4f}%")

    sents, labels = load_data(args.test_file, mode="test")
    for label in labels:
        assert label in label_category
    model.eval()



