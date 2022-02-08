import argparse
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import set_seed, compute_model_size, write_to_file
from data import GEDDataset
from metric import compute_metrics
from collactor import DataCollactorForGED

def train(model, data_loader, optimizer, id2label):
    model.train()

    epoch_loss = 0
    predict_label_lst, label_lst = [], []
    pbar = tqdm(data_loader)
    for i, (inputs, labels, _) in enumerate(pbar):
        inputs["labels"] = labels
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)

        loss = outputs.loss
        loss.backward()

        logits = outputs.logits
        predict_labels = logits.argmax(-1)

        optimizer.step()

        epoch_loss += loss.item()
        label_lst.extend([id2label[label] for label in labels.tolist()])
        predict_label_lst.extend([id2label[label] for label in predict_labels.tolist()])
        pbar.set_description(f"loss={epoch_loss / (i+1):.4f}, acc={compute_metrics(predict_label_lst, label_lst):.4f}")

    return epoch_loss / len(data_loader), compute_metrics(predict_label_lst, label_lst)

def valid(model, data_loader, id2label):
    model.eval()

    epoch_loss = 0
    results = []

    pbar = tqdm(data_loader)
    with torch.no_grad():
        for i, (inputs, labels, inputs_oral) in enumerate(pbar):
            inputs["labels"] = labels
            inputs = inputs.to(device)
            outputs = model(**inputs)

            loss = outputs.loss
            epoch_loss += loss.item()

            logits = outputs.logits
            predict_labels = logits.argmax(-1)

            labels = labels.tolist()
            predict_labels = predict_labels.tolist()
            for input_oral, predict_label, label in zip(inputs_oral, predict_labels, labels):
                results.append([input_oral, id2label[predict_label], id2label[label]])

            predict_label_lst = [result[1] for result in results]
            label_lst = [result[2] for result in results]
            pbar.set_description(f"loss={epoch_loss / (i + 1):.4f}, acc={compute_metrics(predict_label_lst, label_lst):.4f}")

    return epoch_loss / len(data_loader), compute_metrics(predict_label_lst, label_lst), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a classification model to predict if a sentence is grammar true")
    parser.add_argument("--model_name_or_path", default="model/chinese_bert", type=str)
    parser.add_argument("--train_file", default="data/sighan/train.json", type=str)
    parser.add_argument("--valid_file", default="data/sighan/dev.json", type=str)
    parser.add_argument("--output_dir", default="ged_classification/checkpoints/bert", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--valid_batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    label2id = {"T": 0, "F": 1}
    id2label = {value: key for key, value in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2id))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = model.to(device)
    compute_model_size(model)

    train_dataset = GEDDataset(file_path=args.train_file, mode="train")
    valid_dataset = GEDDataset(file_path=args.valid_file, mode="valid")

    collactor = DataCollactorForGED(tokenizer=tokenizer, max_length=args.max_length, label2id=label2id)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collactor, batch_size=args.train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, collate_fn=collactor, batch_size=args.valid_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    patience = 0
    valid_loss, valid_acc, _ = valid(model, valid_dataloader, id2label)
    store_acc = {
        "valid_acc": valid_acc
    }
    print(f"Before training:  valid_loss {valid_loss:.4f};  valid_acc {valid_acc:.4f}")

    for epoch in range(args.epochs):
        print(f"Start train {epoch+1}th epoch")
        train_loss, train_acc = train(model, train_dataloader, optimizer, id2label)
        print(f"Epoch {epoch+1}th:  train_loss {train_loss:.4f};  train_acc {train_acc:.4f}")

        print(f"Start valid {epoch+1}th epoch")
        valid_loss, valid_acc, results = valid(model, valid_dataloader, id2label)
        print(f"Epoch {epoch+1}th:  valid_loss {valid_loss:.4f};  valid_acc {valid_acc:.4f}")

        if valid_acc > store_acc["valid_acc"]:
            store_acc["train_acc"] = train_acc
            store_acc["valid_acc"] = valid_acc
            patience = 0
            torch.save({
                "config": args,
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "label2id": label2id
            }, os.path.join(args.output_dir, "model.tar"))
            print(f"Save model to {args.output_dir}")
            write_to_file(results, os.path.join(args.output_dir, "result.json"))
        else:
            patience += 1
            print(f"patience up to {patience}")

        if patience == args.patience:
            print("Training Over!")
            print(f"Best train_acc {store_acc['train_acc']:.4f}, valid_acc {store_acc['valid_acc']}")
            break

    if patience < args.patience:
        print("Training Over!")
        print(f"Best train_acc {store_acc['train_acc']:.4f}, valid_acc {store_acc['valid_acc']:.4f}")









