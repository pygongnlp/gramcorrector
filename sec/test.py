import argparse
import torch
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import load_data, write_to_file
from metric import compute_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="model/chinese_bert", type=str)
    parser.add_argument("--save_path", default="./", type=str)
    parser.add_argument("--test_file", default="data/sighan/test.json", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    checkpoint = torch.load(os.path.join(args.save_path, "model.tar"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    src, trg = load_data(file_path=args.test_file, mode="test")

    results = []
    for s, t in zip(src, trg):
        inputs = tokenizer(t, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)

        logits = outputs.logits[0][1:-1]  #filter [CLS] & [SEP]
        predict = tokenizer.convert_ids_to_tokens(logits.argmax(-1).tolist())

        s_tok = tokenizer.tokenize(s)
        t_tok = tokenizer.tokenize(t)
        assert len(s_tok) == len(t_tok) == len(predict)
        results.append([s_tok, t_tok, predict])

    metrics = compute_metrics(results)
    print(f"{', '.join([f'{key}={value:.4f}' for key, value in metrics.items()])}")

    write_to_file(file_path=os.path.join(args.save_path, "result_test.json"), results=results)
    print(f"write to {os.path.join(args.save_path, 'result_test.json')}")









