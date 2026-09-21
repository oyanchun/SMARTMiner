import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from SMARTClassifier import SMARTClassifier
from SpanQualifier import SpanQualifier, evaluate


EXTRACT_MODEL_NAME = os.getenv("EXTRACT_MODEL_NAME", "microsoft/deberta-v3-base")
CLASSIFY_MODEL_NAME = os.getenv("CLASSIFY_MODEL_NAME", "microsoft/deberta-v3-large")
EXTRACT_MODEL_PATH = Path(
    os.getenv("EXTRACT_MODEL_PATH", "/app/models/extract/pytorch_model.bin")
)
CLASSIFY_MODEL_PATH = Path(
    os.getenv("CLASSIFY_MODEL_PATH", "/app/models/classify/pytorch_model.bin")
)
MAX_SPAN_GAP = int(os.getenv("MAX_SPAN_GAP", "47"))
EXTRACT_MAX_LEN = int(os.getenv("EXTRACT_MAX_LEN", "512"))
CLASSIFY_MAX_LEN = int(os.getenv("CLASSIFY_MAX_LEN", "64"))
DIM2 = int(os.getenv("DIM2", "64"))
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


class PredictRequest(BaseModel):
    note: str = Field(min_length=1, max_length=100_000)


class ClassifyRequest(BaseModel):
    goals: list[str] = Field(min_length=1, max_length=100)


class SMARTMinerService:
    def __init__(self):
        self.extract_tokenizer = AutoTokenizer.from_pretrained(EXTRACT_MODEL_NAME)
        self.classify_tokenizer = AutoTokenizer.from_pretrained(CLASSIFY_MODEL_NAME)

        self.extract_model = SpanQualifier(
            EXTRACT_MODEL_NAME,
            MAX_SPAN_GAP,
            DIM2,
            EXTRACT_MAX_LEN,
            DEVICE,
        ).to(DEVICE)
        extract_checkpoint = torch.load(EXTRACT_MODEL_PATH, map_location=DEVICE)
        self.extract_model.load_state_dict(
            extract_checkpoint.get("model_state_dict", extract_checkpoint),
            strict=False,
        )
        self.extract_model.eval()

        self.classify_model = SMARTClassifier(CLASSIFY_MODEL_NAME).to(DEVICE)
        classify_checkpoint = torch.load(CLASSIFY_MODEL_PATH, map_location=DEVICE)
        self.classify_model.load_state_dict(
            classify_checkpoint.get("model_state_dict", classify_checkpoint),
            strict=False,
        )
        self.classify_model.float().eval()

    @torch.inference_mode()
    def predict(self, note: str) -> list[str]:
        example = [{
            "id": "request",
            "question": "what are the smart goals mentioned in the text ?",
            "context": note.strip(),
            "answers": [],
            "answers_idx": [],
        }]
        _, results = evaluate(
            self.extract_model,
            example,
            eval_batch_size=1,
            max_len=EXTRACT_MAX_LEN,
            tokenizer=self.extract_tokenizer,
            device=DEVICE,
        )
        return results.get("request", [])

    @torch.inference_mode()
    def classify(self, goals: list[str]) -> list[dict]:
        encoding = self.classify_tokenizer(
            goals,
            truncation=True,
            padding="max_length",
            max_length=CLASSIFY_MAX_LEN,
            return_tensors="pt",
        )
        output = self.classify_model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
        )
        scores = torch.stack(
            [output["specific"], output["measurable"], output["attainable"]],
            dim=1,
        ).cpu()

        results = []
        for goal, goal_scores in zip(goals, scores):
            sma_scores = [float(score) for score in goal_scores]
            sma_pred = [int(score >= 0.5) for score in sma_scores]
            total = sum(sma_pred)
            if total == 3:
                classification = "SMART"
            elif total == 2:
                classification = "Partially SMART"
            else:
                classification = "Not SMART"
            results.append({
                "goal": goal,
                "classification": classification,
                "SMA_pred": sma_pred,
                "SMA_scores": sma_scores,
            })
        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = SMARTMinerService()
    yield


app = FastAPI(title="SMARTMiner Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "device": str(DEVICE)}


@app.post("/predict")
def predict(request: PredictRequest):
    return {"goals": app.state.service.predict(request.note)}


@app.post("/classify")
def classify(request: ClassifyRequest):
    return {"results": app.state.service.classify(request.goals)}
