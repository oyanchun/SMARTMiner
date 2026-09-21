"""Train SMARTClassifier on train_classify.jsonl and valid_classify.jsonl.

This is intended for final production training after model selection. The test
split is deliberately not loaded or evaluated.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from SMARTClassifier import (
    SMARTClassifier,
    SMARTGoalDataset,
    compute_loss,
    load_data,
    set_seed,
)


def train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = compute_loss(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    parser = argparse.ArgumentParser(
        description="Train SMARTClassifier on train + validation data."
    )
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--max_len", default=64, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        help="Fixed number of epochs for final training.",
    )
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--seed", default=30, type=int)
    parser.add_argument("--dataset_dir", default="../SMARTSpan")
    parser.add_argument("--split", default="split_1")
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset_root = Path(args.dataset_dir) / args.split
    train_data = load_data(dataset_root / "train_classify.jsonl")
    valid_data = load_data(dataset_root / "valid_classify.jsonl")
    combined_data = train_data + valid_data

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = SMARTGoalDataset(combined_data, tokenizer, args.max_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SMARTClassifier(args.model_name).to(device)
    model.float()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, data_loader, optimizer, device, epoch)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6)})
        print(f"Epoch {epoch + 1}: train loss = {train_loss:.4f}")

    model_name = args.model_name.rsplit("/", 1)[-1]
    parameter_name = (
        f"lr_{args.lr}_seed_{args.seed}_bs_{args.batch_size}"
        f"_epochs_{args.epochs}"
    )
    output_path = (
        Path(args.output_dir)
        / args.split
        / f"classify_{model_name}"
        / f"full_{parameter_name}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    with open(output_path / "training_log.json", "w", encoding="utf-8") as log_file:
        json.dump(
            {
                "training_data": [
                    "train_classify.jsonl",
                    "valid_classify.jsonl",
                ],
                "test_data_used": False,
                "epochs": args.epochs,
                "history": history,
                "output_model_path": str(model_path),
            },
            log_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved final model to {model_path}")


if __name__ == "__main__":
    main()
