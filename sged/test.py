import argparse
import torch
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_data, write_to_file, label2id, id2label
from metric import compute_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--save_path", default="sged/checkpoints/bert", type=str)
    parser.add_argument("--test_file", default="data/sighan/test.json", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(label2id))

    checkpoint = torch.load(os.path.join(args.save_path, "model.tar"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    src, trg = load_data(file_path=args.test_file, mode="test")

    results = []
    with torch.no_grad():
        for s, t in zip(src, trg):
            inputs = tokenizer(s, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)

            predict = id2label[outputs[0].argmax(-1).item()]
            label = "T" if s == t else "F"

            results.append([s, t, label, predict])

    acc = compute_acc(results)
    print(f"acc={acc:.4f}")

    write_to_file(file_path=os.path.join(args.save_path, "result_test.json"), results=results)
    print(f"write to {os.path.join(args.save_path, 'result_test.json')}")









