import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, DistributedSampler

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from utils import set_seed, compute_model_size, load_data, write_to_file, epoch_time
from data import SpellGECDataset
from metric import compute_metrics
from collactor import DataCollactorForSpellGEC


def train(model, data_loader, optimizer, tokenizer, device, step=500):
    model.train()

    epoch_loss = 0
    results = []
    for i, (src, labels, src_tok, trg_tok) in enumerate(data_loader):
        src["labels"] = labels
        src = src.to(device)

        optimizer.zero_grad()
        outputs = model(**src)

        loss = outputs.loss
        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

        predictions = outputs.logits.argmax(-1).tolist()
        labels = labels.tolist()
        for s, t, label, predict in zip(src_tok, trg_tok, labels, predictions):
            predict = tokenizer.convert_ids_to_tokens([p for p_id, p in enumerate(predict) if label[p_id] != -100])
            assert len(s) == len(t) == len(predict), f"{s}  {t}  {predict}  {len(s)}/{len(t)}/{len(predict)}"
            results.append([s, t, predict])

        if (i + 1) % step == 0:
            metrics = compute_metrics(results)
            print(f"Step {i + 1}, loss={epoch_loss / (i + 1):.4f}, "
                  f"{', '.join([f'{key}={value:.4f}' for key, value in metrics.items()])}")

    return epoch_loss / len(data_loader), compute_metrics(results)


def valid(model, data_loader, tokenizer, device, step=500):
    model.eval()

    epoch_loss = 0
    results = []
    with torch.no_grad():
        for i, (src, labels, src_tok, trg_tok) in enumerate(data_loader):
            src["labels"] = labels
            src = src.to(device)
            outputs = model(**src)

            loss = outputs.loss
            epoch_loss += loss.item()

            predictions = outputs.logits.argmax(-1).tolist()
            labels = labels.tolist()
            for s, t, label, predict in zip(src_tok, trg_tok, labels, predictions):
                predict = tokenizer.convert_ids_to_tokens([p for p_id, p in enumerate(predict) if label[p_id] != -100])
                assert len(s) == len(t) == len(predict), f"{s}  {t}  {predict}  {len(s)}/{len(t)}/{len(predict)}"
                results.append([s, t, predict])

            if (i + 1) % step == 0:
                metrics = compute_metrics(results)
                print(f"Step {i + 1}, loss={epoch_loss / (i + 1):.4f}, "
                      f"{', '.join([f'{key}={value:.4f}' for key, value in metrics.items()])}")

    return epoch_loss / len(data_loader), compute_metrics(results), results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Spell Error Correction")
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--train_file", default="data/sighan/train.json", type=str)
    parser.add_argument("--valid_file", default="data/sighan/dev.json", type=str)
    parser.add_argument("--save_path", default="sec/checkpoints/bert", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--valid_batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--step", default=2, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    args = parser.parse_args()
    print(f"Params={args}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    compute_model_size(model)

    train_dataset = SpellGECDataset(file_path=args.train_file, mode="train")
    valid_dataset = SpellGECDataset(file_path=args.valid_file, mode="valid")

    collactor = DataCollactorForSpellGEC(tokenizer=tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collactor)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.valid_batch_size, collate_fn=collactor)

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
    print("Start valid before training...")
    valid_loss, valid_metrics, _ = valid(model, valid_dataloader, tokenizer, device, args.step)
    store_metrics = {
        "valid_metrics": valid_metrics
    }
    print(f"Before training, valid_loss={valid_loss:.4f}, {', '.join([f'{key}={value:.4f}' for key, value in valid_metrics.items()])}")

    all_start_time = time.time()
    for epoch in range(args.epochs):
        print(f"Start train {epoch + 1}th epochs")
        start_time = time.time()
        train_loss, train_metrics = train(model, train_dataloader, optimizer, tokenizer, device, args.step)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch {epoch + 1}th:  time={epoch_mins}m{epoch_secs}s, "
              f"train_loss={train_loss:.4f}, {', '.join([f'{key}={value:.4f}' for key, value in train_metrics.items()])}")

        print(f"Start valid {epoch + 1}th epochs")
        start_time = time.time()
        valid_loss, valid_metrics, results = valid(model, valid_dataloader, tokenizer, device, args.step)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch {epoch + 1}th:  time={epoch_mins}m{epoch_secs}s, "
              f"valid_loss={valid_loss:.4f}, {', '.join([f'{key}={value:.4f}' for key, value in valid_metrics.items()])}")

        if valid_metrics["cor_f1"] > store_metrics["valid_metrics"]["cor_f1"]:
            store_metrics["train_metrics"] = train_metrics
            store_metrics["valid_metrics"] = valid_metrics
            patience = 0

            torch.save({
                "config": args,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "valid_metrics": valid_metrics,
                "train_metrics": train_metrics,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }, os.path.join(args.save_path, "model.tar"))
            print(f"save model to {args.save_path}")

            write_to_file(os.path.join(args.save_path, "result_valid.json"), results)
            print(f"write result to {os.path.join(args.save_path, 'result_valid.json')}")
        else:
            patience += 1
            print(f"patience up to {patience}")

        if patience == args.patience:
            all_end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(all_start_time, all_end_time)
            print("Training Over!")
            print(f"All time={epoch_mins}m{epoch_secs}s")
            print(
                f"Best train_metrics=({', '.join([f'{key}={value:.4f}' for key, value in store_metrics['train_metrics'].items()])})"
                f"valid_metrics=({', '.join([f'{key}={value:.4f}' for key, value in store_metrics['valid_metrics'].items()])})")
            break

    if patience < args.patience:
        all_end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(all_start_time, all_end_time)
        print("Training Over!")
        print(f"All time={epoch_mins}m{epoch_secs}s")
        print(
            f"Best train_metrics=({', '.join([f'{key}={value:.4f}' for key, value in store_metrics['train_metrics'].items()])})"
            f"valid_metrics=({', '.join([f'{key}={value:.4f}' for key, value in store_metrics['valid_metrics'].items()])})")









