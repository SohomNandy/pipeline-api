"""
Stage 3a — Vulnerability NER
Model: Swapnanil09/vulnerability-extractor
       → microsoft/codebert-base + LoRA adapter (adapter_model.safetensors)
Platform: Render CPU (free tier, 512MB)

THE FIX:
  The HF repo is a PEFT adapter repo (2.42MB adapter_model.safetensors).
  It does NOT contain full model weights — that's why pytorch_model.bin 404'd.
  Correct pattern: load codebert-base, then apply the adapter on top.
  Wrong (original): AutoModelForTokenClassification.from_pretrained(adapter_repo)

Memory on Render 512MB:
  codebert-base float16  ~240MB
  LoRA adapter            ~2.4MB
  tokenizer + runtime     ~80MB
  Total                  ~322MB ✓
"""

import os, re, json, secrets as _secrets, hashlib, logging
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("stage3a")

API_KEY      = os.environ.get("STAGE_3A_API_KEY", "")
PORT         = int(os.environ.get("PORT", "8000"))
BASE_REPO    = "microsoft/codebert-base"
ADAPTER_REPO = "Swapnanil09/vulnerability-extractor"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(API_KEY.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

_tokenizer   = None
_model       = None
_model_ready = False
_load_error  = None
_CVE_RE      = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — PEFT pattern
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    global _tokenizer, _model, _model_ready, _load_error
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN", "") or None

    # Step 1 — tokenizer (lives in adapter repo)
    log.info(f"Loading tokenizer from {ADAPTER_REPO}...")
    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO, token=hf_token)

    # Step 2 — read num_labels + label mappings from adapter config.json
    cfg_path = hf_hub_download(ADAPTER_REPO, "config.json", token=hf_token)
    with open(cfg_path) as f:
        cfg = json.load(f)
    num_labels = cfg.get("num_labels", 17)
    id2label   = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    label2id   = cfg.get("label2id", {})
    log.info(f"  {num_labels} labels: {list(id2label.values())}")

    # Step 3 — base codebert-base in float16
    log.info(f"Loading base model {BASE_REPO} in float16...")
    base = AutoModelForTokenClassification.from_pretrained(
        BASE_REPO,
        num_labels        = num_labels,
        id2label          = id2label,
        label2id          = label2id,
        dtype             = torch.float16,
        low_cpu_mem_usage = True,
        token             = hf_token,
        ignore_mismatched_sizes = True,
    )

    # Step 4 — apply LoRA adapter
    log.info(f"Applying LoRA adapter from {ADAPTER_REPO}...")
    _model = PeftModel.from_pretrained(base, ADAPTER_REPO,
                                       is_trainable=False, token=hf_token)
    _model.eval()
    for p in _model.parameters():
        p.requires_grad = False

    log.info(f"✓ Stage 3a ready — {sum(p.numel() for p in _model.parameters()):,} params")
    _model_ready = True


# ══════════════════════════════════════════════════════════════════════════════
# NER INFERENCE
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
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue

        # CodeBERT is RoBERTa-based — uses Ġ for word-start, no prefix = continuation
        # Also handle ## (WordPiece) defensively
        if token.startswith("##"):
            clean, is_sub = token[2:], True
        elif token.startswith("Ġ"):
            clean, is_sub = token[1:], False
        else:
            clean, is_sub = token, bool(cur and label.startswith("I-"))

        if is_sub and cur:
            cur["text"] += ("" if token.startswith("##") else " ") + clean
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


def _cve_ids(entities, raw_text):
    found = set()
    for e in entities:
        if e["type"] in ("ERROR", "EXPLOIT", "SOFTWARE"):
            found.update(m.upper() for m in _CVE_RE.findall(e["text"]))
    found.update(m.upper() for m in _CVE_RE.findall(raw_text))
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
    global _load_error
    try:
        load_model()
    except Exception as e:
        _load_error = str(e)
        log.error(f"Model load failed: {e}")
    yield

app = FastAPI(title="Stage 3a — Vulnerability NER", version="3.1.0", lifespan=lifespan)

class ExtractRequest(BaseModel):
    log_text:    str
    edge_id:     Optional[str] = ""
    scenario_id: Optional[str] = ""
    t:           Optional[int] = 0

class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]

@app.get("/health")
def health():
    return {
        "stage":       "3a",
        "status":      "ok" if _model_ready else "loading",
        "model_ready": _model_ready,
        "base":        BASE_REPO,
        "adapter":     ADAPTER_REPO,
        "load_error":  _load_error,
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
                "edge_id": r.edge_id, "scenario_id": r.scenario_id, "t": r.t,
                "log_text":  r.log_text[:200],
                "entities":  ents,
                "cve_ids":   cves,
                "software":  grouped.get("SOFTWARE", []),
                "versions":  grouped.get("VERSION",  []),
                "exploits":  grouped.get("EXPLOIT",  []),
                "ips":       grouped.get("IP",       []),
                "users":     grouped.get("USER",     []),
                "ports":     grouped.get("PORT",     []),
                "paths":     grouped.get("PATH",     []),
                "errors":    grouped.get("ERROR",    []),
                "n_entities":len(ents),
            })
        except Exception as e:
            log.warning(f"NER error: {e}")
            results.append({
                "edge_id": r.edge_id, "scenario_id": r.scenario_id, "t": r.t,
                "log_text": r.log_text[:200], "error": str(e),
                "entities":[], "cve_ids":[], "software":[], "versions":[],
                "exploits":[], "ips":[], "users":[], "ports":[], "paths":[],
                "errors":[], "n_entities":0,
            })
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")