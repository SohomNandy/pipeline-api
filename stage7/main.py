"""
Stage 7 — Temporal GRU + Optional Groq Analysis
Model Architecture: GRU(128→128) → Dropout(0.2) → Linear(128→1) → Sigmoid
Input:  h_v from Stage 6, shape (N, 128) — tiled to (N, T, 128) internally
Output: next_step_predictions (N, T), final_hidden (N, 128)
LLM:    Groq llama-3.3-70b-versatile (optional, include_analysis=True)
"""
import os
import time
import random
import modal
import torch
import torch.nn as nn
import hashlib
import secrets as _secrets
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# MODAL APP & IMAGE DEFINITION (✅ FIXES NameError)
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
        "huggingface_hub",
        "groq>=0.9.0",
    )
)

app = modal.App("stage7-temporal-gru", image=image)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REPO_ID = "sohomn/stage7-temporal-gnn"
MODEL_FILE = "model_GRU.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "stage7_gru_weights.pt")

INPUT_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2
DEFAULT_T = int(os.getenv("DEFAULT_TIMESTEPS", "20"))
MAX_GROQ_WORKERS = int(os.getenv("MAX_GROQ_WORKERS", "3"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

USE_GPU = os.getenv("STAGE7_USE_GPU", "1").lower() == "1"
GPU_CONFIG = "T4" if USE_GPU else None
CPU_COUNT = 2 if not USE_GPU else 0

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION (Matches Doc Exactly)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x: torch.Tensor):
        gru_out, h_n = self.gru(x)           # gru_out: (N, T, 128), h_n: (1, N, 128)
        h_n = h_n.squeeze(0)                 # (N, 128)
        dropped = self.dropout(gru_out)
        pred = self.linear(dropped).squeeze(-1)  # (N, T)
        return torch.sigmoid(pred), h_n

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT LOADER (Robust Fallback)
# ─────────────────────────────────────────────────────────────────────────────
def load_model_weights(model: TemporalGRU) -> bool:
    """Try HF → Local → Random Init. Returns True if trained weights loaded."""
    try:
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID, filename=MODEL_FILE, token=os.getenv("HF_TOKEN")
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[INFO] Loaded trained weights from {REPO_ID}/{MODEL_FILE}")
        return True
    except Exception as e:
        print(f"[WARN] HF load failed: {e}")

    if os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"[INFO] Loaded weights from local path: {WEIGHTS_PATH}")
            return True
        except Exception as e:
            print(f"[WARN] Local load failed: {e}")

    print("[WARN] No trained weights found. Initializing with random weights for testing.")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# MODAL CLASS
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    cpu=CPU_COUNT,
    gpu=GPU_CONFIG,
    memory=4096 if GPU_CONFIG else 2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class TemporalGRUPredictor:
    def __enter__(self):
        self.model = TemporalGRU()
        self.has_trained_weights = load_model_weights(self.model)
        self.model.eval()

        groq_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.groq_available = self.groq_client is not None

    @modal.method()
    def predict(self, h_v: List[List[float]], n_timesteps: int = DEFAULT_T, include_analysis: bool = False) -> dict:
        x = torch.tensor(h_v, dtype=torch.float32)
        x = x.unsqueeze(1).expand(-1, n_timesteps, -1)

        with torch.no_grad():
            preds, final_h = self.model(x)

        preds_list = preds.tolist()
        final_h_list = final_h.tolist()

        analysis = None
        if include_analysis and self.groq_available:
            batch_meta = [
                {"idx": i, "max_prob": float(max(preds_list[i])), "h_norm": float(torch.norm(torch.tensor(final_h_list[i])))}
                for i in range(len(h_v))
            ]
            analysis = self._groq_analyze(batch_meta)
        elif include_analysis:
            analysis = ["Groq analysis skipped: GROQ_API_KEY not configured."] * len(h_v)

        return {
            "next_step_predictions": preds_list,
            "final_hidden": final_h_list,
            "n_nodes": len(h_v),
            "n_timesteps": n_timesteps,
            "analysis": analysis,
            "weights_status": "trained" if self.has_trained_weights else "random_init"
        }

    def _groq_analyze(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        def _worker(data: Dict[str, Any]) -> str:
            prompt = (
                f"Node {data['idx']} temporal risk profile:\n"
                f"Peak compromise probability: {data['max_prob']:.4f}\n"
                f"Final hidden state L2 norm: {data['h_norm']:.4f}\n"
                f"Provide: 1) Risk level (Low/Med/High/Critical) 2) One-sentence mitigation recommendation."
            )
            for attempt in range(4):
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150, temperature=0.2, timeout=10.0
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = (2 ** attempt) + random.uniform(0, 1.5)
                        time.sleep(wait)
                    else:
                        return f"Analysis failed: {str(e)}"
            return "Analysis failed: exhausted retries"

        results = [None] * len(batch_data)
        with ThreadPoolExecutor(max_workers=MAX_GROQ_WORKERS) as executor:
            future_to_idx = {executor.submit(_worker, item): i for i, item in enumerate(batch_data)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Analysis failed: {str(e)}"
        return results

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
@app.function(cpu=CPU_COUNT, gpu=GPU_CONFIG, memory=4096 if GPU_CONFIG else 2048, secrets=[modal.Secret.from_name("siem-pipeline-secrets")], container_idle_timeout=300)
@modal.asgi_app()
def fastapi_app():
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.getenv("STAGE_7_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    class PredictRequest(BaseModel):
        h_v: List[List[float]] = Field(..., description="Shape: (N, 128) from Stage 6")
        scenario_id: str
        n_nodes: int
        n_timesteps: int = Field(DEFAULT_T, ge=1, le=50)
        include_analysis: bool = Field(False, description="Run Groq LLM analysis")

    class PredictResponse(BaseModel):
        next_step_predictions: List[List[float]]
        final_hidden: List[List[float]]
        n_nodes: int
        n_timesteps: int
        scenario_id: str
        analysis: Optional[List[str]] = None
        metadata: Dict[str, Any] = {}

    web = FastAPI(title="Stage 7 — Temporal GRU")
    predictor = TemporalGRUPredictor()

    @web.get("/health")
    async def health():
        return {
            "stage": "7",
            "status": "ok",
            "model": f"{REPO_ID}/{MODEL_FILE}",
            "groq_model": GROQ_MODEL,
            "weights_loaded": predictor.has_trained_weights
        }

    @web.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, _=Depends(validate)):
        if not req.h_v:
            raise HTTPException(status_code=400, detail="h_v is empty")
        if len(req.h_v[0]) != INPUT_DIM:
            raise HTTPException(status_code=400, detail=f"h_v embedding dim must be {INPUT_DIM}, got {len(req.h_v[0])}")

        result = predictor.predict.remote(
            h_v=req.h_v,
            n_timesteps=req.n_timesteps,
            include_analysis=req.include_analysis,
        )

        return PredictResponse(
            next_step_predictions=result["next_step_predictions"],
            final_hidden=result["final_hidden"],
            n_nodes=result["n_nodes"],
            n_timesteps=result["n_timesteps"],
            scenario_id=req.scenario_id,
            analysis=result["analysis"],
            metadata={
                "groq_model": GROQ_MODEL,
                "groq_workers": MAX_GROQ_WORKERS,
                "analysis_enabled": req.include_analysis,
                "weights_status": result["weights_status"]
            }
        )

    return web