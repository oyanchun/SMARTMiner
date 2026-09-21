import argparse, os, torch, json
from pathlib import Path
from transformers import AutoTokenizer
from SpanQualifier import SpanQualifier, evaluate
from SMARTClassifier import SMARTClassifier
from utils import read_dataset

# Extract goals from one free-text health coaching note.
def extract_goals(
    text,
    output_model_path=None,
    model_name="microsoft/deberta-v3-base",
    max_span_gap=47,
    dim2=64,
    max_len=512,
    device=None,
    classify_model_path=None,
    classify_model_name="microsoft/deberta-v3-large",
    classify_max_len=64,
):
    """Extract goals and classify each one as SMART, partially SMART, or not SMART."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    if output_model_path is None:
        output_model_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "split_1"
            / "extract_deberta-v3-base"
            / "lr_3e-05_seed_30_bs_32_ga_4"
            / "pytorch_model.bin"
        )
    output_model_path = Path(output_model_path)
    if not output_model_path.is_file():
        raise FileNotFoundError(f"SpanQualifier checkpoint not found: {output_model_path}")

    checkpoint = torch.load(output_model_path, map_location=device)
    model = SpanQualifier(model_name, max_span_gap, dim2, max_len, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    example = [{
        "id": "free_text",
        "question": "what are the smart goals mentioned in the text ?",
        "context": text.strip(),
        "answers": [],
        "answers_idx": [],
    }]
    _, results = evaluate(
        model,
        example,
        eval_batch_size=1,
        max_len=max_len,
        tokenizer=tokenizer,
        device=device,
    )
    goals = results.get("free_text", [])
    if not goals:
        return []

    if classify_model_path is None:
        classify_model_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "split_1"
            / "classify_deberta-v3-large"
            / "lr_2e-05_seed_30_bs_4"
            / "pytorch_model.bin"
        )
    classify_model_path = Path(classify_model_path)
    if not classify_model_path.is_file():
        raise FileNotFoundError(
            f"SMARTClassifier checkpoint not found: {classify_model_path}"
        )

    classifier_tokenizer = AutoTokenizer.from_pretrained(classify_model_name)
    classifier = SMARTClassifier(classify_model_name).to(device)
    classifier.float()
    classifier.load_state_dict(
        torch.load(classify_model_path, map_location=device)
    )
    classifier.eval()

    classified_goals = []
    with torch.no_grad():
        for goal in goals:
            encoding = classifier_tokenizer(
                goal,
                truncation=True,
                padding="max_length",
                max_length=classify_max_len,
                return_tensors="pt",
            )
            output = classifier(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
            sma_scores = [
                float(output["specific"].item()),
                float(output["measurable"].item()),
                float(output["attainable"].item()),
            ]
            sma_pred = [int(score >= 0.5) for score in sma_scores]
            total = sum(sma_pred)
            if total == 3:
                classification = "SMART"
            elif total == 2:
                classification = "Partially SMART"
            else:
                classification = "Not SMART"

            classified_goals.append({
                "goal": goal,
                "classification": classification,
                "SMA_pred": sma_pred,
                "SMA_scores": sma_scores,
            })

    return classified_goals

# inference for classification
def classifying_goals(model_name, output_model_path, goals_dic_data, device, max_lenght):

    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SMARTClassifier(model_name).to(device)
    model.float()
    print(f"Loading best model from {output_model_path}")
    model.load_state_dict(torch.load(output_model_path))
    model.to(device)
    
    with torch.no_grad():
        for hc_id in goals_dic_data.keys():     
            for goal in goals_dic_data.get(hc_id):

                encoding = tokenizer(goal, truncation=True, padding="max_length",
                                max_length=max_lenght, return_tensors="pt")
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                
                sma_pred = [
                    int(output["specific"]   >= 0.5),
                    int(output["measurable"] >= 0.5),
                    int(output["attainable"] >= 0.5),
                ]
                
                def to_label(sma):
                    total = sum(sma)
                    if total == 3:
                        return "SMART"
                    elif total == 2:
                        return "Partially SMART"
                    else:
                        return "Not SMART"

                final_pred = to_label(sma_pred)

                results.append({
                        "id": hc_id,
                        "extracted_goal": goal,
                        "predicted": final_pred,
                        "SMA_pred": sma_pred
                    })

    return results

# function for extracting goals
def goal_extraction(output_model_path, device, extract_model_name,max_span_gap, dim2, 
                extract_max_len, dataset_dir):

    checkpoint = torch.load(output_model_path, map_location=device)
    model = SpanQualifier(extract_model_name, max_span_gap, dim2, extract_max_len, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    data_path_test = f"{dataset_dir}/{args.split}/test.json"
    test_examples = read_dataset(data_path_test)

    tokenizer = AutoTokenizer.from_pretrained(extract_model_name)
    _, results = evaluate(model, test_examples, eval_batch_size=1, 
                        max_len=extract_max_len, tokenizer=tokenizer, device=device)

    return results

# main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu", 
                        default="0")
    parser.add_argument("--extract_model_name", 
                        default="microsoft/deberta-v3-base")
    parser.add_argument("--classify_model_name", 
                        default="microsoft/deberta-v3-large")

    parser.add_argument("--output_dir",
                        default="../outputs",
                        help="Path to model checkpoints.")
    parser.add_argument("--dataset_dir",
                        default="../SMARTSpan",
                        type=str,
                        help="The data directory where data splits are found.")
    parser.add_argument("--results_dir",
                        default="../results",
                        type=str)
    parser.add_argument("--split",
                        default="split_1",
                        type=str,
                        help="Data split.")

    parser.add_argument("--extract_bs",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--extract_as",
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--extract_max_len",
                        default=512,
                        type=int)
    parser.add_argument("--extract_lr",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dim2",
                        default=64,
                        type=int)
    parser.add_argument("--max_span_gap", 
                        type=int, 
                        default=47, 
                        help="Value used during training")

    parser.add_argument("--classify_lr",
                        default=2e-5,
                        type=float,
                        help="Learning rate for classification")
    parser.add_argument("--classify_bs",
                        default=4,
                        type=int,
                        help="Batch size for training/evaluation")
    parser.add_argument("--classify_max_len",
                        default=64,
                        type=int,
                        help="Maximum sequence length")
    
    parser.add_argument("--seed",
                        type=int,
                        default=30,
                        help="random seed for initialization")
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # device = torch.device(f"cuda:{args.gpu}")
    device = torch.device("cpu")

    # prepare paths for goal extraction
    extract_model_name_abb = args.extract_model_name.split("/")[-1]
    extract_parameter_name = f"lr_{args.extract_lr}_seed_{args.seed}_bs_{args.extract_bs}" \
                      f"_ga_{args.extract_as}"
    extract_output_model_path = f"{args.output_dir}/{args.split}/extract_{extract_model_name_abb}/{extract_parameter_name}/pytorch_model.bin"

    # extract goals from HC notes
    extracted_goals = goal_extraction(extract_output_model_path, device, args.extract_model_name, 
                                      args.max_span_gap, args.dim2, args.extract_max_len, 
                                      args.dataset_dir)
    
    # prepare paths for goal classification
    classify_model_name_abb = args.classify_model_name.split("/")[-1]
    classify_parameter_name = f"lr_{args.classify_lr}_seed_{args.seed}_bs_{args.classify_bs}"
    classify_output_model_path = f"{args.output_dir}/{args.split}/classify_{classify_model_name_abb}/{classify_parameter_name}/pytorch_model.bin"

    # classifying goals from HC notes
    results = classifying_goals(args.classify_model_name, classify_output_model_path, 
                      extracted_goals, device, args.classify_max_len)
    
    # write predictions
    path_save_final_result = f"{args.results_dir}/{args.split}/pipeline/"
    os.makedirs(path_save_final_result, exist_ok=True)

    with open(f"{path_save_final_result}/test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
