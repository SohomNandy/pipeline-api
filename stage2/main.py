import os, sys, torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 2 - Log Embeddings")
validate = get_api_key_validator("2")
_model   = None


def get_model():
    global _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        tok  = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        mdl  = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        mdl.eval()
        proj = torch.nn.Linear(1024, 256, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        _model = {"tok": tok, "mdl": mdl, "proj": proj}
    return _model


@app.on_event("startup")
async def startup():
    get_model()


class EmbedRequest(BaseModel):
    entity_id: str
    log_texts: List[str]


@app.get("/health")
async def health():
    return {
        "stage": "2",
        "status": "stub",
        "note": "LoRA adapters not yet trained - using base BGE-Large",
    }


@app.post("/embed")
async def embed(req: EmbedRequest, _=Depends(validate)):
    m      = get_model()
    text   = "query: " + " [SEP] ".join(req.log_texts[:20])
    inputs = m["tok"](text, return_tensors="pt", truncation=True,
                      max_length=512, padding=True)
    with torch.no_grad():
        out  = m["mdl"](**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb  = torch.nn.functional.normalize(emb, dim=-1)
        z    = m["proj"](emb)[0]
    return {
        "entity_id": req.entity_id,
        "z_log":     z.tolist(),
        "dim":       256,
        "_stub":     True,
    }
