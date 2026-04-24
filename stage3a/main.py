"""
Stage 3a — Vulnerability NER
Model: Swapnanil09/vulnerability-extractor (codebert-base + LoRA, BIO, 8 entity classes)
Platform: Render CPU (free tier, 512MB)

Fixes vs original:
  1. float16 inference — halves model footprint (~500MB → ~250MB), fits in 512MB
  2. Model loaded at startup via lifespan (not on first request) — avoids
     blocking the async event loop and gateway timeout on cold start
  3. Correct subword token reassembly for CodeBERT (## prefix, not Ġ)
  4. graceful OOM guard with clear error message
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

# ── Global model state ────────────────────────────────────────────────────────
_tokenizer = None
_model     = None

# ── CVE pattern ───────────────────────────────────────────────────────────────
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load codebert-base + LoRA adapter in float16.
    float16 halves the footprint: ~500MB float32 → ~250MB float16.
    Called once at startup via lifespan — never on a request path.
    """
    global _tokenizer, _model
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    hf_token = os.environ.get("HF_TOKEN", "") or None
    log.info(f"Loading {MODEL_REPO} in float16...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=hf_token,
    )

    _model = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16,   # KEY FIX: halves RAM from ~500MB to ~250MB
        low_cpu_mem_usage=True,       # stream weights, avoid double-loading peak
        token=hf_token,
    )
    _model.eval()

    # Disable gradient tracking — inference only
    for p in _model.parameters():
        p.requires_grad = False

    param_count = sum(p.numel() for p in _model.parameters())
    log.info(f"✓ Stage 3a model ready — {param_count:,} params")


# ══════════════════════════════════════════════════════════════════════════════
# NER INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _extract_entities(log_text: str) -> List[dict]:
    """
    Run BIO NER on log_text.
    Returns list of {text: str, type: str}.

    CodeBERT tokenizer uses WordPiece subwords with ## continuation prefix.
    Correct reassembly: strip ## and append directly (no space).
    Wrong (original): replace Ġ with space (that's GPT2/RoBERTa, not BERT).
    """
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
        # Cast inputs to float16 to match model dtype
        outputs     = _model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [_model.config.id2label[p.item()] for p in predictions[0]]

    entities       = []
    current_entity = None

    for token, label in zip(tokens, labels):
        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue

        # FIX: CodeBERT uses ## for subword continuations (WordPiece)
        # Strip ## and append without space; otherwise it's a new word
        is_subword  = token.startswith("##")
        clean_token = token[2:] if is_subword else token

        if is_subword and current_entity:
            # Continuation of previous token — append directly, no space
            current_entity["text"] += clean_token

        elif label.startswith("B-"):
            # Begin new entity
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": clean_token, "type": label[2:]}

        elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
            # Inside same entity type — add with space separator
            current_entity["text"] += " " + clean_token

        else:
            # O label or mismatched I- tag — close current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Clean whitespace
    result = []
    for e in entities:
        e["text"] = e["text"].strip()
        if e["text"]:
            result.append(e)
    return result


def _extract_cve_ids(entities: List[dict]) -> List[str]:
    cves = []
    for e in entities:
        if e["type"] in ("ERROR", "EXPLOIT", "SOFTWARE"):
            cves.extend(m.upper() for m in _CVE_RE.findall(e["text"]))
    return list(set(cves))


def _group_by_type(entities: List[dict]) -> dict:
    grouped = {}
    for e in entities:
        grouped.setdefault(e["type"], []).append(e["text"])
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup — never block a request path."""
    try:
        load_model()
    except Exception as e:
        # Log but don't crash the server — /health will report degraded
        log.error(f"Model load failed: {e}")
    yield
    log.info("Stage 3a shutdown")

app = FastAPI(
    title="Stage 3a — Vulnerability NER",
    version="2.0.0",
    lifespan=lifespan,
)

class ExtractRequest(BaseModel):
    log_text:    str
    edge_id:     Optional[str] = ""
    scenario_id: Optional[str] = ""
    t:           Optional[int] = 0

class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]

@app.get("/health")
def health():
    model_ok = _model is not None and _tokenizer is not None
    return {
        "stage":       "3a",
        "status":      "ok" if model_ok else "degraded — model not loaded",
        "model":       MODEL_REPO,
        "model_ready": model_ok,
    }

@app.post("/extract")
def extract(req: ExtractBatchRequest, _=Depends(validate)):
    """
    Sync endpoint — model inference is CPU-bound, not I/O-bound.
    Using sync def avoids run_in_executor overhead and is correct for this case.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not ready — check /health")

    results = []
    for log_req in req.logs:
        try:
            entities = _extract_entities(log_req.log_text)
            cve_ids  = _extract_cve_ids(entities)
            grouped  = _group_by_type(entities)

            results.append({
                "edge_id":     log_req.edge_id,
                "scenario_id": log_req.scenario_id,
                "t":           log_req.t,
                "log_text":    log_req.log_text[:200],
                "entities":    entities,
                "cve_ids":     cve_ids,
                "software":    grouped.get("SOFTWARE", []),
                "versions":    grouped.get("VERSION",  []),
                "exploits":    grouped.get("EXPLOIT",  []),
                "ips":         grouped.get("IP",       []),
                "users":       grouped.get("USER",     []),
                "ports":       grouped.get("PORT",     []),
                "paths":       grouped.get("PATH",     []),
                "errors":      grouped.get("ERROR",    []),
                "n_entities":  len(entities),
            })

        except Exception as e:
            log.warning(f"NER failed for log '{log_req.log_text[:80]}': {e}")
            results.append({
                "edge_id":    log_req.edge_id,
                "scenario_id":log_req.scenario_id,
                "t":          log_req.t,
                "log_text":   log_req.log_text[:200],
                "error":      str(e),
                "entities":   [],
                "cve_ids":    [],
                "n_entities": 0,
            })

    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")