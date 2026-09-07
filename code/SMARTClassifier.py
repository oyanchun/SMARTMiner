import json, torch, random, argparse, os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import classification_report


def report_to_dict(report):
    return {
        key: {metric: float(value) for metric, value in metrics.items()}
        if isinstance(metrics, dict) else float(metrics)
        for key, metrics in report.items()
    }
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# class for dataset
class SMARTGoalDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        goal = self.data[idx]["goal"]
        label = torch.tensor(self.data[idx]["label"], dtype=torch.float)
        encoded = self.tokenizer(goal, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label
        }

# class for model
class SMARTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.head_s = nn.Linear(hidden_size, 1)
        self.head_m = nn.Linear(hidden_size, 1)
        self.head_a = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.relu(self.linear(self.dropout(x)))
        return {
            "specific": self.sigmoid(self.head_s(x)).squeeze(-1),
            "measurable": self.sigmoid(self.head_m(x)).squeeze(-1),
            "attainable": self.sigmoid(self.head_a(x)).squeeze(-1)
        }

# loss function
def compute_loss(preds, labels):
    bce = nn.BCELoss()
    return (bce(preds["specific"], labels[:, 0]) +
            bce(preds["measurable"], labels[:, 1]) +
            bce(preds["attainable"], labels[:, 2])) / 3

# final labels: SMART/Partially SMART/Not SMART
def get_final_label(batch_preds, threshold=0.5):
    final = []
    for s, m, a in zip(batch_preds["specific"], batch_preds["measurable"], batch_preds["attainable"]):
        total = int(s >= threshold) + int(m >= threshold) + int(a >= threshold)
        if total == 3:
            final.append("SMART")
        elif total == 2:
            final.append("Partially SMART")
        else:
            final.append("Not SMART")
    return final

# evaluation function
def evaluate(model, dataloader, name="valid", raw_data=None, res_path=None, device=None):
    model.eval()
    gold, pred = [], []
    total_loss = 0
    results = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask)
            loss = compute_loss(preds, labels)
            total_loss += loss.item()

            batch_gold = get_final_label({
                "specific": labels[:, 0],
                "measurable": labels[:, 1],
                "attainable": labels[:, 2]
            })
            batch_pred = get_final_label(preds)

            gold += batch_gold
            pred += batch_pred

            if res_path and raw_data:
                for j in range(len(batch_gold)):
                    raw_idx = i * dataloader.batch_size + j
                    raw_entry = raw_data[raw_idx]
                    pred_sma = [
                        int(preds["specific"][j].item() >= 0.5),
                        int(preds["measurable"][j].item() >= 0.5),
                        int(preds["attainable"][j].item() >= 0.5)
                    ]
                    results.append({
                        "id": raw_entry.get("id", f"{i}_{j}"),
                        "goal": raw_entry["goal"],
                        "ground_truth": batch_gold[j],
                        "predicted": batch_pred[j],
                        "SMA_gold": [int(x) for x in raw_entry["label"]],
                        "SMA_pred": pred_sma
                    })

    report = classification_report(gold, pred, digits=3, output_dict=True)
    report_text = classification_report(gold, pred, digits=3)
    print(f"\n{name} classification report: {report_text}")

    # saving predictions and stats
    if res_path:
        os.makedirs(res_path, exist_ok=True)

        # write predictions
        with open(f"{res_path}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # write stats
        with open(f"{res_path}/{name}_stats.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        # write structured report
        with open(f"{res_path}/{name}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return total_loss / len(dataloader)

# train loop
def train_loop(model, train_loader, valid_loader, optimizer, model_path, epochs, device, patience, results_dir):
    best_val_loss = float('inf')
    best_epoch = None
    best_val_report = None
    best_test_report = None
    patience_counter = 0
    training_log = []
    model.float()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = compute_loss(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train loss = {avg_train_loss:.4f}")

        val_loss = evaluate(model, valid_loader, name="Validation", res_path=results_dir, device=device)
        print(f"Epoch {epoch+1}: Validation loss = {val_loss:.4f} (best so far: {best_val_loss:.4f})")

        val_report_path = os.path.join(results_dir, "Validation_report.json")
        val_report = None
        if os.path.exists(val_report_path):
            with open(val_report_path, "r", encoding="utf-8") as f:
                val_report = json.load(f)

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_report": val_report,
            "is_best": False,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")

            best_val_report = val_report
            epoch_log["is_best"] = True
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                training_log.append(epoch_log)
                break

        training_log.append(epoch_log)

    with open(os.path.join(results_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
            "best_val_result": best_val_report,
            "history": training_log,
        }, f, ensure_ascii=False, indent=2)
        
# function for fine tuning
def train(model, tokenizer, data_set_dir, max_len, batch_size, device, lr, 
        output_model_path, epochs, patience, results_dir):

    train_data = load_data(os.path.join(data_set_dir, "train_classify.jsonl"))
    valid_data = load_data(os.path.join(data_set_dir, "valid_classify.jsonl"))
    
    train_set = SMARTGoalDataset(train_data, tokenizer, max_len)
    valid_set = SMARTGoalDataset(valid_data, tokenizer, max_len)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size) 
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    train_loop(model, train_loader, valid_loader, optimizer, output_model_path, epochs, device, 
            patience, results_dir)

# main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu",
                        default="0",
                        type=str,
                        help="GPU id(s) to use, e.g. '0' or '0,1'")
    parser.add_argument("--model_name",
                        default="microsoft/deberta-v3-large",
                        type=str,
                        help="Name of base model")
    parser.add_argument("--only_eval",
                        default=False,
                        help="Run only evaluation (skip training)")
    parser.add_argument("--max_len",
                        default=64,
                        type=int,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size for training/evaluation")
    parser.add_argument("--epochs",
                        default=20,
                        type=int,
                        help="Number of training epochs")
    parser.add_argument("--patience",
                        default=5,
                        type=int,
                        help="Early stopping patience")
    parser.add_argument("--lr",
                        default=2e-5,
                        type=float,
                        help="Learning rate")
    parser.add_argument("--seed",
                        default=30,
                        type=int,
                        help="Random seed")
    parser.add_argument("--dataset_dir",
                        default="../SMARTSpan",
                        type=str,
                        help="The data directory where data splits are found.")
    parser.add_argument("--split",
                        default="split_1",
                        type=str,
                        help="Data split.")
    parser.add_argument("--output_dir",
                        default="../outputs",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--results_dir",
                        default="../results",
                        type=str)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # device = torch.device(f"cuda:{args.gpu}")
    device = torch.device("cpu")

    model_name_abb = args.model_name.split("/")[-1]
    parameter_name = f"lr_{args.lr}_seed_{args.seed}_bs_{args.batch_size}"

    dataset_path = f"{args.dataset_dir}/{args.split}/"
    output_model_path = f"{args.output_dir}/{args.split}/classify_{model_name_abb}/{parameter_name}/pytorch_model.bin"
    path_save_result = f"{args.results_dir}/{args.split}/classify_{model_name_abb}/"

    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SMARTClassifier(args.model_name).to(device)

    # Training
    if args.only_eval == False:
        train(model, tokenizer, dataset_path, args.max_len, args.batch_size, device,
                args.lr, output_model_path, args.epochs, args.patience, path_save_result)

    # Evaluation
    print(f"\nLoading best model from {output_model_path}")
    model.load_state_dict(torch.load(output_model_path))
    model.to(device)

    test_data  = load_data(os.path.join(dataset_path, "test_classify.jsonl"))
    test_set  = SMARTGoalDataset(test_data, tokenizer, args.max_len)
    test_loader  = DataLoader(test_set, args.batch_size)

    evaluate(model, test_loader, name="test", raw_data=test_data, res_path=path_save_result, 
             device=device)
