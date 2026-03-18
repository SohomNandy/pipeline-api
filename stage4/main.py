import os, sys, hashlib
from fastapi import FastAPI, Depends
from pydantic import BaseModel
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 4 - Cross-Cloud Identity Embeddings")
validate = get_api_key_validator("4")


class IdentityRequest(BaseModel):
    identity: str
    provider: str


@app.get("/health")
async def health():
    return {"stage": "4", "status": "stub", "note": "Contrastive model not yet trained"}


@app.post("/embed")
async def embed(req: IdentityRequest, _=Depends(validate)):
    key  = f"{req.identity}@{req.provider}"
    h    = hashlib.sha256(key.encode()).digest()
    vec  = [(b / 127.5) - 1.0 for b in h]
    z_id = (vec * 4)[:128]
    norm = sum(x ** 2 for x in z_id) ** 0.5
    z_id = [x / norm for x in z_id]
    return {
        "identity":   req.identity,
        "provider":   req.provider,
        "z_identity": z_id,
        "dim":        128,
        "_stub":      True,
    }
