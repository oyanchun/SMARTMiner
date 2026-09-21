"""Train the SpanQualifier extractor on train.json and valid.json combined.

This is intended for the final production-training pass after model selection.
The test set is deliberately not loaded or evaluated by this script.
"""

import argparse
import ast
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import trange

import SpanQualifier as span_qualifier
from SpanQualifier import SpanQualifier, get_input_feature
from utils import read_dataset, save_model, set_seed


def load_initial_checkpoint(model, checkpoint_path):
    raw = torch.load(checkpoint_path, map_location="cpu")
    state = raw.get("model_state_dict", raw)
    new_state_dict = OrderedDict()

    for key, value in state.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module."):]
        normalized_key = normalized_key.replace("bert.bert.", "bert.")
        if (
            normalized_key.startswith(
                ("bert.", "deberta.", "roberta.", "encoder.", "transformer.")
            )
            and not normalized_key.startswith("token_representation.")
        ):
            normalized_key = "token_representation." + normalized_key
        new_state_dict[normalized_key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Initialized from: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")


def train_epoch(
    model,
    examples,
    tokenizer,
    optimizer,
    scheduler,
    batch_size,
    gradient_accumulation_steps,
    max_len,
    device,
    epoch,
):
    model.train()
    model.float()
    model.to(device)
    order = list(range(len(examples)))
    random.shuffle(order)
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    batch_count = (len(order) + batch_size - 1) // batch_size
    progress = trange(batch_count, desc=f"Epoch {epoch + 1}")
    for batch_number in progress:
        start = batch_number * batch_size
        batch_examples = [examples[index] for index in order[start:start + batch_size]]
        input_ids, token_type_ids, attention_mask, context_ranges, _, _, targets = (
            get_input_feature(
                batch_examples,
                max_source_length=max_len,
                tokenizer=tokenizer,
                device=device,
            )
        )

        loss = model(
            input_ids,
            token_type_ids,
            attention_mask,
            context_ranges,
            targets=targets,
        ).mean()
        total_loss += loss.item()
        steps += 1

        (loss / gradient_accumulation_steps).backward()
        is_last_batch = batch_number == batch_count - 1
        if (
            (batch_number + 1) % gradient_accumulation_steps == 0
            or is_last_batch
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix(loss=f"{total_loss / steps:.4f}")

    return total_loss / steps


def main():
    parser = argparse.ArgumentParser(
        description="Train SpanQualifier on train.json + valid.json."
    )
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer_name", default="microsoft/deberta-v3-base")
    parser.add_argument(
        "--init_checkpoint",
        default=True,
        type=ast.literal_eval,
        help="Initialize the encoder from the MultiSpanQA checkpoint.",
    )
    parser.add_argument(
        "--hf_repo_id_fine_tuned",
        default="ivabojic/deberta-v3-base_MultiSpanQA",
    )
    parser.add_argument("--hf_filename", default="pytorch_model.bin")
    parser.add_argument("--local_init_checkpoint", default=None)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--dim2", default=64, type=int)
    parser.add_argument("--max_span_gap", default=47, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        help="Fixed number of epochs for final training.",
    )
    parser.add_argument("--seed", default=30, type=int)
    parser.add_argument("--dataset_dir", default="../SMARTSpan")
    parser.add_argument("--split", default="split_1")
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be at least 1")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    span_qualifier.device = device
    set_seed(args.seed)

    dataset_root = Path(args.dataset_dir) / args.split
    train_examples = read_dataset(dataset_root / "train.json")
    valid_examples = read_dataset(dataset_root / "valid.json")
    combined_examples = train_examples + valid_examples

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = SpanQualifier(
        args.model_name,
        args.max_span_gap,
        args.dim2,
        args.max_len,
        device,
    ).to(device)

    if args.init_checkpoint:
        if args.local_init_checkpoint:
            checkpoint_path = args.local_init_checkpoint
        else:
            checkpoint_path = hf_hub_download(
                repo_id=args.hf_repo_id_fine_tuned,
                filename=args.hf_filename,
            )
        load_initial_checkpoint(model, checkpoint_path)

    effective_batch_size = max(
        1, args.train_batch_size // args.gradient_accumulation_steps
    )
    batch_count = (
        len(combined_examples) + effective_batch_size - 1
    ) // effective_batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_updates = (
        (batch_count + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps
    ) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_updates),
        num_training_steps=total_updates,
    )

    model_name = args.model_name.rsplit("/", 1)[-1]
    parameter_name = (
        f"lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}"
        f"_ga_{args.gradient_accumulation_steps}_epochs_{args.epochs}"
    )
    output_path = (
        Path(args.output_dir)
        / args.split
        / f"extract_{model_name}"
        / f"full_{parameter_name}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            combined_examples,
            tokenizer,
            optimizer,
            scheduler,
            effective_batch_size,
            args.gradient_accumulation_steps,
            args.max_len,
            device,
            epoch,
        )
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6)})
        print(f"Epoch {epoch + 1}: train loss = {train_loss:.4f}")

    save_model(str(output_path) + os.sep, model, optimizer)
    with open(output_path / "training_log.json", "w", encoding="utf-8") as log_file:
        json.dump(
            {
                "training_data": ["train.json", "valid.json"],
                "test_data_used": False,
                "epochs": args.epochs,
                "max_span_gap": args.max_span_gap,
                "best_epoch": None,
                "history": history,
                "output_model_path": str(output_path / "pytorch_model.bin"),
            },
            log_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved final model to {output_path / 'pytorch_model.bin'}")


if __name__ == "__main__":
    main()
