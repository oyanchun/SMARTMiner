import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from SMARTClassifier import SMARTClassifier, SMARTGoalDataset, evaluate as evaluate_classifier, load_data
from SpanQualifier import SpanQualifier, evaluate as evaluate_span, prettyprint
from utils import read_dataset


def write_text_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_extractor_final(
    model_path,
    split,
    dataset_dir,
    model_name="microsoft/deberta-v3-base",
    max_len=512,
    dim2=64,
    max_span_gap=47,
    eval_batch_size=1,
    device=None,
    results_dir=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)
    test_examples = read_dataset(dataset_dir / split / "test.json")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = SpanQualifier(model_name, max_span_gap, dim2, max_len, device).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()

    result_score, results = evaluate_span(
        model,
        test_examples,
        eval_batch_size=eval_batch_size,
        max_len=max_len,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"[Extractor] split={split} model={model_path.name}")
    print(f"[Extractor] metrics: {prettyprint(result_score)}")
    if results_dir is not None:
        out_dir = Path(results_dir) / split / "final_eval" / "extract"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(result_score, f, ensure_ascii=False, indent=2)
        write_text_report(
            out_dir / "test_metrics.txt",
            [
                f"SMARTMiner SpanQualifier evaluation",
                f"Split: {split}",
                f"Model checkpoint: {model_path}",
                f"Test set: {dataset_dir / split / 'test.json'}",
                "",
                f"Exact-match F1: {result_score['em_f1']:.2f}",
                f"Overlap F1: {result_score['overlap_f1']:.2f}",
            ],
        )
    return result_score, results


def evaluate_classifier_final(
    model_path,
    split,
    dataset_dir,
    model_name="microsoft/deberta-v3-large",
    max_len=64,
    batch_size=4,
    device=None,
    results_dir=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)
    test_data = load_data(dataset_dir / split / "test_classify.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = SMARTClassifier(model_name).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()

    test_set = SMARTGoalDataset(test_data, tokenizer, max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    eval_loss = evaluate_classifier(
        model,
        test_loader,
        name="test",
        raw_data=test_data,
        res_path=str(Path(results_dir) / split / "final_eval" / "classify") if results_dir is not None else None,
        device=device,
    )

    print(f"[Classifier] split={split} model={model_path.name}")
    print(f"[Classifier] test loss: {eval_loss:.4f}")
    if results_dir is not None:
        out_dir = Path(results_dir) / split / "final_eval" / "classify"
        write_text_report(
            out_dir / "test_metrics.txt",
            [
                "SMARTMiner SMARTClassifier evaluation",
                f"Split: {split}",
                f"Model checkpoint: {model_path}",
                f"Test set: {dataset_dir / split / 'test_classify.jsonl'}",
                "",
                f"Test loss: {eval_loss:.4f}",
                "Detailed classification metrics: test_stats.txt",
            ],
        )
    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate final SMARTMiner models on the split test set.")
    parser.add_argument("--dataset_dir", default="../SMARTSpan", type=str)
    parser.add_argument("--split", default="split_1", type=str)
    parser.add_argument("--results_dir", default="../results", type=str)
    parser.add_argument("--gpu", default="0", type=str)

    parser.add_argument("--extract_model_path", type=str, default=None)
    parser.add_argument("--extract_model_name", default="microsoft/deberta-v3-base", type=str)
    parser.add_argument("--extract_max_len", default=512, type=int)
    parser.add_argument("--extract_eval_batch_size", default=1, type=int)
    parser.add_argument("--dim2", default=64, type=int)
    parser.add_argument("--max_span_gap", default=47, type=int)

    parser.add_argument("--classify_model_path", type=str, default=None)
    parser.add_argument("--classify_model_name", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--classify_max_len", default=64, type=int)
    parser.add_argument("--classify_batch_size", default=4, type=int)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor_result = None
    classifier_loss = None
    if args.extract_model_path:
        extractor_result, _ = evaluate_extractor_final(
            model_path=args.extract_model_path,
            split=args.split,
            dataset_dir=args.dataset_dir,
            model_name=args.extract_model_name,
            max_len=args.extract_max_len,
            dim2=args.dim2,
            max_span_gap=args.max_span_gap,
            eval_batch_size=args.extract_eval_batch_size,
            device=device,
            results_dir=args.results_dir,
        )

    if args.classify_model_path:
        classifier_loss = evaluate_classifier_final(
            model_path=args.classify_model_path,
            split=args.split,
            dataset_dir=args.dataset_dir,
            model_name=args.classify_model_name,
            max_len=args.classify_max_len,
            batch_size=args.classify_batch_size,
            device=device,
            results_dir=args.results_dir,
        )

    if not args.extract_model_path and not args.classify_model_path:
        raise ValueError("Provide at least one of --extract_model_path or --classify_model_path")

    summary_dir = Path(args.results_dir) / args.split / "final_eval"
    summary_lines = [
        "SMARTMiner final-model evaluation",
        f"Split: {args.split}",
        f"Device: {device}",
        "",
    ]
    if extractor_result is not None:
        summary_lines.extend(
            [
                "SpanQualifier:",
                f"  Checkpoint: {args.extract_model_path}",
                f"  Exact-match F1: {extractor_result['em_f1']:.2f}",
                f"  Overlap F1: {extractor_result['overlap_f1']:.2f}",
                "",
            ]
        )
    if classifier_loss is not None:
        summary_lines.extend(
            [
                "SMARTClassifier:",
                f"  Checkpoint: {args.classify_model_path}",
                f"  Test loss: {classifier_loss:.4f}",
                "  Detailed classification metrics: classify/test_stats.txt",
                "",
            ]
        )
    write_text_report(summary_dir / "evaluation_summary.txt", summary_lines)


if __name__ == "__main__":
    main()
