import os, sys, torch, torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Optional
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 4 - Cross-Cloud Identity Embeddings")
validate = get_api_key_validator("4")
_model   = None

REPO_ID = "sohomn/stage4-identity-embeddings"
OUT_DIM = 128


def get_model():
    global _model
    if _model is not None:
        return _model

    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Loading {REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID, token=hf_token or None
    )
    base    = AutoModel.from_pretrained("google/flan-t5-base")
    encoder = PeftModel.from_pretrained(
        base, REPO_ID + "/adapter", token=hf_token or None
    )
    encoder.eval()

    # Load projection head — weights_only=True avoids arbitrary code execution
    proj_path = hf_hub_download(REPO_ID, "proj_head.pt", token=hf_token or None)
    proj = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256, 128)
    )
    proj.load_state_dict(
        torch.load(proj_path, map_location="cpu", weights_only=True)
    )
    proj.eval()

    _model = {"tokenizer": tokenizer, "encoder": encoder, "proj": proj}
    print("✓ Stage 4 model ready")
    return _model


def embed_identity(identity: str, provider: str) -> List[float]:
    m         = get_model()
    text      = f"identity: {identity} provider: {provider}"
    inputs    = m["tokenizer"](
        text, return_tensors="pt", max_length=32,
        truncation=True, padding=True
    )
    with torch.no_grad():
        out  = m["encoder"].encoder(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        z    = m["proj"](emb)
        z    = F.normalize(z, dim=-1)
    return z[0].tolist()


class IdentityRequest(BaseModel):
    identity:    str
    provider:    str
    entity_type: Optional[str] = "User"


class IdentityBatchRequest(BaseModel):
    identities: List[IdentityRequest]


@app.get("/health")
async def health():
    return {
        "stage":  "4",
        "status": "ok",
        "model":  REPO_ID,
        "out_dim": OUT_DIM,
        "note":   "model loads on first request (~20s cold start)",
    }


@app.post("/embed")
async def embed(req: IdentityRequest, _=Depends(validate)):
    return {
        "identity":   req.identity,
        "provider":   req.provider,
        "z_identity": embed_identity(req.identity, req.provider),
        "dim":        OUT_DIM,
    }


@app.post("/embed_batch")
async def embed_batch(req: IdentityBatchRequest, _=Depends(validate)):
    results = [
        {
            "identity":   r.identity,
            "provider":   r.provider,
            "z_identity": embed_identity(r.identity, r.provider),
            "dim":        OUT_DIM,
        }
        for r in req.identities
    ]
    return {"embeddings": results, "total": len(results)}