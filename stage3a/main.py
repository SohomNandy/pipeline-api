"""
Stage 3a — Vulnerability NER
Model: Swapnanil09/vulnerability-extractor
Platform: Render CPU (free tier, 512MB)

ROOT CAUSE OF ALL PREVIOUS 503s — TWO ISSUES:
  1. GATED MODEL: The HF repo requires accepting access conditions.
     Render never has a valid HF_TOKEN with accepted terms → model load fails
     silently → _model stays None → every request returns 503.
     FIX: HF_TOKEN env var must be set in Render, AND Swapnanil must accept
     the gating on his own HF account (Settings → Access Requests).

  2. WRONG LOADING PATTERN: Previous code used PeftModel.from_pretrained()
     treating it as adapter-only. The model card shows plain
     AutoModelForTokenClassification.from_pretrained() — it's a merged model.
     adapter_config.json exists but the weights are merged into one file.
     FIX: Use the exact loading pattern from the model card.

  3. WARMUP LOOP: Service now blocks startup until model is fully loaded.
     /health returns model_ready:false during load, true when ready.
     main_flow.py polls /health via the gateway before sending batches.
     FIX: Added /stage3a/health route to gateway main.py.

MEMORY:
  codebert-base full model float16 = ~240MB
  Runtime overhead               = ~80MB
  Total                          = ~320MB ✓ fits in 512MB
"""

import os, re, secrets as _secrets, hashlib, logging, threading
from contextlib import asynccontextmanager
from typing import List, Optional

import torch, uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("stage3a")

API_KEY      = os.environ.get("STAGE_3A_API_KEY", "")
PORT         = int(os.environ.get("PORT", "8000"))
MODEL_REPO   = "Swapnanil09/vulnerability-extractor"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(API_KEY.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# ── Global model state ────────────────────────────────────────────────────────
_tokenizer   = None
_model       = None
_model_ready = False
_load_error  = None
_load_lock   = threading.Lock()

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# Loading pattern taken exactly from the model card.
# The repo is a merged model (not adapter-only despite adapter_config.json).
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    global _tokenizer, _model, _model_ready, _load_error

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    hf_token = os.environ.get("HF_TOKEN", "") or None
    if not hf_token:
        log.warning("HF_TOKEN not set — gated model will fail to load. "
                    "Set HF_TOKEN in Render env vars.")

    log.info(f"Loading tokenizer from {MODEL_REPO}...")
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=hf_token,
    )

    log.info(f"Loading model from {MODEL_REPO} in float16...")
    # Plain from_pretrained — matches model card exactly.
    # dtype= instead of torch_dtype= (transformers ≥4.40 deprecation).
    _model = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO,
        dtype             = torch.float16,
        low_cpu_mem_usage = True,
        token             = hf_token,
    )
    _model.eval()
    for p in _model.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in _model.parameters())
    log.info(f"✓ Stage 3a ready — {n:,} params | labels: {list(_model.config.id2label.values())}")
    _model_ready = True


def _load_in_background():
    """Load model in a background thread so lifespan doesn't block server startup."""
    global _load_error
    with _load_lock:
        try:
            load_model()
        except Exception as e:
            _load_error = str(e)
            log.error(f"Model load failed: {e}")
            log.error("Check: 1) HF_TOKEN is set in Render env vars")
            log.error("       2) Swapnanil accepted gating at huggingface.co/Swapnanil09/vulnerability-extractor")


# ══════════════════════════════════════════════════════════════════════════════
# NER INFERENCE
# Token reassembly uses Ġ (RoBERTa/CodeBERT BPE) — confirmed from model card.
# ══════════════════════════════════════════════════════════════════════════════

def _extract_entities(text: str) -> List[dict]:
    if not _model_ready:
        raise RuntimeError("Model not loaded")

    inputs = _tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=128, padding=True)

    with torch.no_grad():
        preds = torch.argmax(_model(**inputs).logits, dim=-1)

    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [_model.config.id2label[p.item()] for p in preds[0]]

    entities, cur = [], None
    for token, label in zip(tokens, labels):
        # RoBERTa special tokens
        if token in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"):
            continue

        # RoBERTa BPE: Ġ = word boundary (new word), no prefix = continuation
        if token.startswith("Ġ"):
            clean, is_cont = token[1:], False
        elif token.startswith("##"):      # WordPiece fallback
            clean, is_cont = token[2:], True
        else:
            # No prefix: continuation if we're inside an entity and label is I-
            clean = token
            is_cont = (cur is not None and label.startswith("I-"))

        if is_cont and cur:
            cur["text"] += clean
        elif label.startswith("B-"):
            if cur: entities.append(cur)
            cur = {"text": clean, "type": label[2:]}
        elif label.startswith("I-") and cur and label[2:] == cur["type"]:
            cur["text"] += " " + clean
        else:
            if cur: entities.append(cur)
            cur = None

    if cur:
        entities.append(cur)

    return [{"text": e["text"].strip(), "type": e["type"]}
            for e in entities if e["text"].strip()]


def _cve_ids(entities, raw):
    found = set()
    for e in entities:
        if e["type"] in ("ERROR", "EXPLOIT", "SOFTWARE"):
            found.update(m.upper() for m in _CVE_RE.findall(e["text"]))
    found.update(m.upper() for m in _CVE_RE.findall(raw))
    return list(found)

def _group(entities):
    g = {}
    for e in entities:
        g.setdefault(e["type"], []).append(e["text"])
    return g


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    # Start model loading in background thread immediately
    t = threading.Thread(target=_load_in_background, daemon=True)
    t.start()
    log.info("Model loading started in background...")
    yield

app = FastAPI(title="Stage 3a — Vulnerability NER", version="3.2.0", lifespan=lifespan)

class ExtractRequest(BaseModel):
    log_text:    str
    edge_id:     Optional[str] = ""
    scenario_id: Optional[str] = ""
    t:           Optional[int] = 0

class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]


@app.get("/health")
def health():
    """
    Polled by main_flow.py warmup loop via gateway /stage3a/health.
    Returns model_ready:true only when model is fully loaded.
    """
    return {
        "stage":       "3a",
        "status":      "ok" if _model_ready else "loading",
        "model_ready": _model_ready,
        "model":       MODEL_REPO,
        "load_error":  _load_error,
        "tip":         None if _model_ready else (
            "Model loading — takes ~3min on cold start. "
            "If load_error is set, check HF_TOKEN and gating acceptance."
        ),
    }


@app.post("/extract")
def extract(req: ExtractBatchRequest, _=Depends(validate)):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready — check /health")

    results = []
    for r in req.logs:
        try:
            ents    = _extract_entities(r.log_text)
            cves    = _cve_ids(ents, r.log_text)
            grouped = _group(ents)
            results.append({
                "edge_id":    r.edge_id,  "scenario_id": r.scenario_id, "t": r.t,
                "log_text":   r.log_text[:200],
                "entities":   ents,       "cve_ids":  cves,
                "software":   grouped.get("SOFTWARE", []),
                "versions":   grouped.get("VERSION",  []),
                "exploits":   grouped.get("EXPLOIT",  []),
                "ips":        grouped.get("IP",       []),
                "users":      grouped.get("USER",     []),
                "ports":      grouped.get("PORT",     []),
                "paths":      grouped.get("PATH",     []),
                "errors":     grouped.get("ERROR",    []),
                "n_entities": len(ents),
            })
        except Exception as e:
            log.warning(f"NER error: {e}")
            results.append({
                "edge_id": r.edge_id, "scenario_id": r.scenario_id, "t": r.t,
                "log_text": r.log_text[:200], "error": str(e),
                "entities":[], "cve_ids":[], "software":[], "versions":[],
                "exploits":[], "ips":[], "users":[], "ports":[], "paths":[],
                "errors":[], "n_entities": 0,
            })
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")