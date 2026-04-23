"""
Stage 3a — Vulnerability NER
Model: Swapnanil09/vulnerability-extractor
Platform: Render CPU (512MB)

Fixes applied:
- Proper model readiness tracking (_MODEL_READY)
- Warmup inference pass (forces lazy init completion)
- Accurate /health endpoint (not misleading)
- Correct gating for /extract (prevents false 503 loops)
"""

import os
import re
import secrets as _secrets
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("stage3a")

API_KEY    = os.environ.get("STAGE_3A_API_KEY", "")
PORT       = int(os.environ.get("PORT", "8000"))
MODEL_REPO = "Swapnanil09/vulnerability-extractor"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(API_KEY.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

_tokenizer = None
_model     = None
_MODEL_READY = False
_MODEL_WARMING = False


_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model():
    global _tokenizer, _model, _MODEL_READY, _MODEL_WARMING

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    hf_token = os.environ.get("HF_TOKEN", "") or None

    log.info(f"Loading {MODEL_REPO} in float16...")

    _MODEL_WARMING = True
    _MODEL_READY = False

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=hf_token,
    )

    _model = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    _model.eval()

    for p in _model.parameters():
        p.requires_grad = False

    # 🔥 CRITICAL FIX: warmup inference pass
    try:
        dummy = _tokenizer("system warmup test", return_tensors="pt")
        with torch.no_grad():
            _model(**dummy)
        log.info("Warmup inference successful")
    except Exception as e:
        log.warning(f"Warmup failed (non-fatal): {e}")

    _MODEL_READY = True
    _MODEL_WARMING = False

    param_count = sum(p.numel() for p in _model.parameters())
    log.info(f"✓ Stage 3a READY — {param_count:,} params")


# ─────────────────────────────────────────────────────────────
# NER LOGIC
# ─────────────────────────────────────────────────────────────

def _extract_entities(log_text: str) -> List[dict]:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded")

    inputs = _tokenizer(
        log_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [_model.config.id2label[p.item()] for p in predictions[0]]

    entities = []
    current = None

    for token, label in zip(tokens, labels):
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue

        is_subword = token.startswith("##")
        clean = token[2:] if is_subword else token

        if is_subword and current:
            current["text"] += clean

        elif label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"text": clean, "type": label[2:]}

        elif label.startswith("I-") and current and label[2:] == current["type"]:
            current["text"] += " " + clean

        else:
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    return [e for e in entities if e["text"].strip()]


def _extract_cve_ids(entities: List[dict]) -> List[str]:
    cves = []
    for e in entities:
        if e["type"] in ("ERROR", "EXPLOIT", "SOFTWARE"):
            cves.extend(_CVE_RE.findall(e["text"]))
    return list(set([c.upper() for c in cves]))


def _group_by_type(entities: List[dict]) -> dict:
    grouped = {}
    for e in entities:
        grouped.setdefault(e["type"], []).append(e["text"])
    return grouped


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        log.error(f"Model load failed: {e}")
    yield
    log.info("Stage 3a shutdown")


app = FastAPI(
    title="Stage 3a — Vulnerability NER",
    version="2.1.0",
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    log_text: str
    edge_id: Optional[str] = ""
    scenario_id: Optional[str] = ""
    t: Optional[int] = 0


class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]


# ─────────────────────────────────────────────────────────────
# HEALTH (FIXED)
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "stage": "3a",
        "model_loaded": _model is not None,
        "model_ready": _MODEL_READY,
        "warming": _MODEL_WARMING,
        "status": (
            "ready" if _MODEL_READY
            else "warming" if _MODEL_WARMING
            else "not_loaded"
        ),
    }


# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE (FIXED GATING)
# ─────────────────────────────────────────────────────────────

@app.post("/extract")
def extract(req: ExtractBatchRequest, _=Depends(validate)):
    if not _MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Stage 3a model warming — retry shortly"
        )

    results = []

    for log_req in req.logs:
        try:
            entities = _extract_entities(log_req.log_text)
            cve_ids  = _extract_cve_ids(entities)
            grouped  = _group_by_type(entities)

            results.append({
                "edge_id": log_req.edge_id,
                "scenario_id": log_req.scenario_id,
                "t": log_req.t,
                "log_text": log_req.log_text[:200],
                "entities": entities,
                "cve_ids": cve_ids,
                "software": grouped.get("SOFTWARE", []),
                "versions": grouped.get("VERSION", []),
                "exploits": grouped.get("EXPLOIT", []),
                "ips": grouped.get("IP", []),
                "users": grouped.get("USER", []),
                "ports": grouped.get("PORT", []),
                "paths": grouped.get("PATH", []),
                "errors": grouped.get("ERROR", []),
                "n_entities": len(entities),
            })

        except Exception as e:
            log.warning(f"NER failed: {e}")
            results.append({
                "edge_id": log_req.edge_id,
                "scenario_id": log_req.scenario_id,
                "t": log_req.t,
                "log_text": log_req.log_text[:200],
                "error": str(e),
                "entities": [],
                "cve_ids": [],
                "n_entities": 0,
            })

    return {"results": results, "total": len(results)}


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


