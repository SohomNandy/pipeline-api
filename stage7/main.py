"""
Stage 7 — Temporal GRU + Groq Analysis
Model: sohomn/stage7-temporal-gnn/model_GRU.pt
Platform: Modal CPU
Input:  h_v from Stage 6, shape (N, 128) — tiled to (N, T, 128) internally
Output: next_step_predictions (N, T), final_hidden (N, 128)
LLM:    Groq llama-3.3-70b-versatile (optional, include_analysis=True)
"""

import modal, os, time, random
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════════
# MODAL IMAGE + APP
# ══════════════════════════════════════════════════════════════════════════════

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "torch>=2.0.0",
        "numpy>=1.26.0",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
        "groq>=0.9.0",
    )
)

app = modal.App("stage7-temporal-gru", image=image)

REPO_ID    = "sohomn/stage7-temporal-gnn"
MODEL_FILE = "model_GRU.pt"

# Architecture constants
INPUT_DIM      = 128
HIDDEN_DIM     = 128
DROPOUT        = 0.2
DEFAULT_T      = int(os.environ.get("DEFAULT_TIMESTEPS", "20"))
MAX_GROQ_WORKERS = int(os.environ.get("MAX_GROQ_WORKERS", "3"))
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ══════════════════════════════════════════════════════════════════════════════
# GRU PREDICTOR  (weights loaded once in __enter__)
# ══════════════════════════════════════════════════════════════════════════════

@app.cls(
    cpu=2,
    memory=2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class TemporalGRUPredictor:

    def __enter__(self):
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download

        # ── Model definition ──────────────────────────────────────────────────
        class TemporalGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru     = nn.GRU(INPUT_DIM, HIDDEN_DIM, batch_first=True)
                self.dropout = nn.Dropout(DROPOUT)
                self.linear  = nn.Linear(HIDDEN_DIM, 1)

            def forward(self, x):
                # x: (N, T, 128)
                gru_out, h_n = self.gru(x)           # (N, T, 128), (1, N, 128)
                h_n          = h_n.squeeze(0)         # (N, 128)
                dropped      = self.dropout(gru_out)
                pred         = self.linear(dropped).squeeze(-1)  # (N, T)
                return torch.sigmoid(pred), h_n

        # ── Load weights from HuggingFace ─────────────────────────────────────
        print(f"Loading {REPO_ID}/{MODEL_FILE}...")
        ckpt_path = hf_hub_download(
            repo_id  = REPO_ID,
            filename = MODEL_FILE,
            token    = os.environ.get("HF_TOKEN") or None,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        self.model = TemporalGRU()
        # support both raw state_dict and {"model_state_dict": ...} checkpoints
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✓ Stage 7 GRU loaded — {sum(p.numel() for p in self.model.parameters()):,} params")

        # ── Groq client ───────────────────────────────────────────────────────
        from groq import Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=groq_key if groq_key else None)
        self.groq_available = bool(groq_key)

    @modal.method()
    def predict(
        self,
        h_v:             List[List[float]],   # (N, 128) from Stage 6
        n_timesteps:     int = DEFAULT_T,
        include_analysis: bool = False,
    ) -> dict:
        import torch

        # Tile h_v across T timesteps: (N, 128) → (N, T, 128)
        x = torch.tensor(h_v, dtype=torch.float32)          # (N, 128)
        x = x.unsqueeze(1).expand(-1, n_timesteps, -1)      # (N, T, 128)

        with torch.no_grad():
            preds, final_h = self.model(x)

        preds_list   = preds.tolist()    # (N, T)
        final_h_list = final_h.tolist()  # (N, 128)

        analysis = None
        if include_analysis:
            if not self.groq_available:
                analysis = ["Groq unavailable: GROQ_API_KEY not set"] * len(h_v)
            else:
                batch_meta = [
                    {
                        "idx":      i,
                        "max_prob": float(max(preds_list[i])),
                        "h_norm":   float(torch.norm(torch.tensor(final_h_list[i])).item()),
                    }
                    for i in range(len(h_v))
                ]
                analysis = self._groq_analyze(batch_meta)

        return {
            "next_step_predictions": preds_list,   # (N, T) — S_temporal for Stage 8
            "final_hidden":          final_h_list, # (N, 128)
            "n_nodes":               len(h_v),
            "n_timesteps":           n_timesteps,
            "analysis":              analysis,
        }

    def _groq_analyze(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """Parallel Groq analysis with exponential backoff + jitter."""
        def _worker(data: Dict[str, Any]) -> str:
            prompt = (
                f"Node {data['idx']} temporal risk profile:\n"
                f"Peak compromise probability: {data['max_prob']:.4f}\n"
                f"Final hidden state L2 norm: {data['h_norm']:.4f}\n"
                f"Provide: 1) Risk level (Low/Med/High/Critical) "
                f"2) One-sentence mitigation recommendation."
            )
            for attempt in range(4):
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.2,
                        timeout=10.0,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = (2 ** attempt) + random.uniform(0, 1.5)
                        print(f"[GROQ] Rate limited — backoff {wait:.2f}s (attempt {attempt+1})")
                        time.sleep(wait)
                    else:
                        return f"Analysis failed: {e}"
            return "Analysis failed: exhausted retries"

        results = [None] * len(batch_data)
        with ThreadPoolExecutor(max_workers=MAX_GROQ_WORKERS) as executor:
            future_to_idx = {executor.submit(_worker, item): i for i, item in enumerate(batch_data)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Analysis failed: {e}"
        return results


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI WRAPPER  (mirrors Stage 6 pattern)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    cpu=2,
    memory=2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import secrets as _secrets
    import hashlib
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel, Field

    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_7_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    # ── schemas ───────────────────────────────────────────────────────────────
    class PredictRequest(BaseModel):
        # Gateway sends h_v as a flat (N, 128) list — we tile internally
        h_v:              List[List[float]] = Field(..., description="Shape: (N, 128) from Stage 6")
        scenario_id:      str
        n_nodes:          int
        n_timesteps:      int   = Field(DEFAULT_T, ge=1, le=50)
        include_analysis: bool  = Field(False, description="Run Groq LLM analysis")

    class PredictResponse(BaseModel):
        next_step_predictions: List[List[float]]
        final_hidden:          List[List[float]]
        n_nodes:               int
        n_timesteps:           int
        scenario_id:           str
        analysis:              Optional[List[str]] = None
        metadata:              Dict[str, Any]      = {}

    # ── app + predictor ───────────────────────────────────────────────────────
    web       = FastAPI(title="Stage 7 — Temporal GRU")
    predictor = TemporalGRUPredictor()

    @web.get("/health")
    async def health():
        return {
            "stage":      "7",
            "status":     "ok",
            "model":      f"{REPO_ID}/{MODEL_FILE}",
            "groq_model": GROQ_MODEL,
        }

    @web.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, _=Depends(validate)):
        if len(req.h_v) == 0:
            raise HTTPException(status_code=400, detail="h_v is empty")
        if len(req.h_v[0]) != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"h_v embedding dim must be {INPUT_DIM}, got {len(req.h_v[0])}"
            )

        result = predictor.predict.remote(
            h_v              = req.h_v,
            n_timesteps      = req.n_timesteps,
            include_analysis = req.include_analysis,
        )

        return PredictResponse(
            next_step_predictions = result["next_step_predictions"],
            final_hidden          = result["final_hidden"],
            n_nodes               = result["n_nodes"],
            n_timesteps           = result["n_timesteps"],
            scenario_id           = req.scenario_id,
            analysis              = result["analysis"],
            metadata              = {
                "groq_model":       GROQ_MODEL,
                "groq_workers":     MAX_GROQ_WORKERS,
                "analysis_enabled": req.include_analysis,
            },
        )

    return web