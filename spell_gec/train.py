import argparse
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from utils import set_seed, compute_model_size, load_data, write_to_file
from data import SpellGECDataset
from metric import compute_metrics
from collactor import DataCollactorForSpellGEC

def train(model, data_loader, optimizer, tokenizer):
    model.train()

    epoch_loss = 0
    results = []
    pbar = tqdm(data_loader)
    for i, (src, trg, src_oral, trg_oral) in enumerate(pbar):
        src["labels"] = trg
        src = src.to(device)

        optimizer.zero_grad()
        output = model(**src)

        loss = output.loss
        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

        predictions = output.logits.argmax(-1)
        for s, t, t_l, predict in zip(src_oral, trg_oral, trg, predictions):
            predict = tokenizer.convert_ids_to_tokens([p for p_id, p in enumerate(predict) if t_l[p_id] != -100])
            assert len(s) == len(t) == len(predict), f"{s}  {t}  {predict}  {len(s)}/{len(t)}/{len(predict)}"
            results.append([s, t, predict])

        metrics = compute_metrics(results)
        pbar.set_description(f"loss={epoch_loss / (i + 1):.4f}, "
                             f"{', '.join([f'{key}:{value:.4f}' for key, value in metrics.items()])}")
    return epoch_loss / len(data_loader), compute_metrics(results)

def valid(model, data_loader, tokenizer):
    model.eval()

    epoch_loss = 0
    results = []
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for i, (src, trg, src_oral, trg_oral) in enumerate(pbar):
            src["labels"] = trg
            src = src.to(device)
            output = model(**src)

            loss = output.loss
            epoch_loss += loss.item()

            predictions =output.logits.argmax(-1)
            for s, t, t_l, predict in zip(src_oral, trg_oral, trg, predictions):
                predict = tokenizer.convert_ids_to_tokens([p for p_id, p in enumerate(predict) if t_l[p_id] != -100])
                assert len(s) == len(t) == len(predict), f"{s}  {t}  {predict}  {len(s)}/{len(t)}/{len(predict)}"
                results.append([s, t, predict])

            metrics = compute_metrics(results)
            pbar.set_description(f"loss={epoch_loss / (i + 1):.4f}, "
                                 f"{', '.join([f'{key}:{value:.4f}' for key, value in metrics.items()])}")
    return epoch_loss / len(data_loader), compute_metrics(results), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a spell error correction model")
    parser.add_argument("--model_name_or_path", default="model/chinese_bert", type=str)
    parser.add_argument("--train_file", default="data/sighan/train.json", type=str)
    parser.add_argument("--valid_file", default="data/sighan/dev.json", type=str)
    parser.add_argument("--output_dir", default="spell_gec/checkpoints/bert", type=str)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
    model = model.to(device)
    compute_model_size(model)

    train_dataset = SpellGECDataset(file_path=args.train_file, mode="train")
    valid_dataset = SpellGECDataset(file_path=args.valid_file, mode="valid")

    collactor = DataCollactorForSpellGEC(tokenizer=tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collactor,
                                  batch_size=args.train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, collate_fn=collactor,
                                  batch_size=args.valid_batch_size)

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
    valid_loss, valid_metrics, _ = valid(model, valid_dataloader, tokenizer)
    store_metrics = {
        "valid_metrics": valid_metrics
    }
    print(f"Before training:  valid_loss={valid_loss:.4f}, {', '.join([f'{key}:{value:.4f}' for key, value in valid_metrics.items()])}")

    for epoch in range(args.epochs):
        print(f"Start train {epoch+1}th epochs")
        train_loss, train_metrics = train(model, train_dataloader, optimizer, tokenizer)
        print(f"Epoch {epoch+1}th:  train_loss {train_loss:.4f}, {', '.join([f'{key}:{value:.4f}' for key, value in train_metrics.items()])}")

        print(f"Start valid {epoch+1}th epochs")
        valid_loss, valid_metrics, results = valid(model, valid_dataloader, tokenizer)
        print(f"Epoch {epoch+1}th:  valid_loss={valid_loss:.4f}, {', '.join([f'{key}:{value:.4f}' for key, value in valid_metrics.items()])}")

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
            }, os.path.join(args.output_dir, "model.tar"))
            print(f"save model to {args.output_dir}")

            write_to_file(os.path.join(args.output_dir, "result.json"), results)
        else:
            patience += 1
            print(f"patience up to {patience}")

        if patience == args.patience:
            print("Training Over!")
            print(f"Best train metrics={', '.join([f'{key}:{value:.4f}' for key, value in store_metrics['train_metrics'].items()])}"
                  f"valid metrics={', '.join([f'{key}:{value:.4f}' for key, value in store_metrics['valid_metrics'].items()])}")
            break
    if patience < args.patience:
        print("Training Over!")
        print(f"Best train metrics={', '.join([f'{key}:{value:.4f}' for key, value in store_metrics['train_metrics'].items()])}"
            f"valid metrics={', '.join([f'{key}:{value:.4f}' for key, value in store_metrics['valid_metrics'].items()])}")









