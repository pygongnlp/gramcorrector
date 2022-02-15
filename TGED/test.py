import argparse
import torch
import os

from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import load_data, write_to_file
from metric import compute_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--save_path", default="TGED/checkpoints/bert", type=str)
    parser.add_argument("--test_file", default="data/sighan/test.json", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id = {"T": 0, "F": 1}
    id2label = {value: key for key, value in label2id.items()}
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    # checkpoint = torch.load(os.path.join(args.save_path, "model.tar"), map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model = model.to(device)

    src, trg = load_data(file_path=args.test_file, mode="test")

    results = []
    for s, t in zip(src, trg):
        inputs = tokenizer(t, is_split_into_words=True, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)

        logits = outputs.logits[0][1:-1]  #filter [CLS] & [SEP]
        predict = [id2label[i] for i in logits.argmax(-1).tolist()]
        assert len(s) == len(t) == len(predict), f"{s}  {t} {predict}"
        results.append([s, t, predict])

    metrics = compute_metrics(results)
    print(f"{', '.join([f'{key}={value:.4f}' for key, value in metrics.items()])}")

    write_to_file(file_path=os.path.join(args.save_path, "result_test.json"), results=results)
    print(f"write to {os.path.join(args.save_path, 'result_test.json')}")









