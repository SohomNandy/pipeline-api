# import os, sys, torch, torch.nn as nn
# import torch.nn.functional as F
# from fastapi import FastAPI, Depends
# from pydantic import BaseModel
# from typing import List, Optional
# sys.path.append("..")
# from shared.auth import get_api_key_validator

# app      = FastAPI(title="Stage 4 - Cross-Cloud Identity Embeddings")
# validate = get_api_key_validator("4")
# _model   = None

# REPO_ID = "sohomn/stage4-identity-embeddings"
# OUT_DIM = 128


# def get_model():
#     global _model
#     if _model is not None:
#         return _model

#     from transformers import AutoTokenizer, AutoModel
#     from peft import PeftModel
#     from huggingface_hub import hf_hub_download

#     hf_token = os.environ.get("HF_TOKEN", "")

#     print(f"Loading {REPO_ID}...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         REPO_ID, token=hf_token or None
#     )
#     base    = AutoModel.from_pretrained("google/flan-t5-base")
#     encoder = PeftModel.from_pretrained(
#         base, REPO_ID + "/adapter", token=hf_token or None
#     )
#     encoder.eval()

#     # Load projection head — weights_only=True avoids arbitrary code execution
#     proj_path = hf_hub_download(REPO_ID, "proj_head.pt", token=hf_token or None)
#     proj = nn.Sequential(
#         nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256, 128)
#     )
#     proj.load_state_dict(
#         torch.load(proj_path, map_location="cpu", weights_only=True)
#     )
#     proj.eval()

#     _model = {"tokenizer": tokenizer, "encoder": encoder, "proj": proj}
#     print("✓ Stage 4 model ready")
#     return _model


# def embed_identity(identity: str, provider: str) -> List[float]:
#     m         = get_model()
#     text      = f"identity: {identity} provider: {provider}"
#     inputs    = m["tokenizer"](
#         text, return_tensors="pt", max_length=32,
#         truncation=True, padding=True
#     )
#     with torch.no_grad():
#         out  = m["encoder"].encoder(**inputs)
#         mask = inputs["attention_mask"].unsqueeze(-1).float()
#         emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#         z    = m["proj"](emb)
#         z    = F.normalize(z, dim=-1)
#     return z[0].tolist()


# class IdentityRequest(BaseModel):
#     identity:    str
#     provider:    str
#     entity_type: Optional[str] = "User"


# class IdentityBatchRequest(BaseModel):
#     identities: List[IdentityRequest]


# @app.get("/health")
# async def health():
#     return {
#         "stage":  "4",
#         "status": "ok",
#         "model":  REPO_ID,
#         "out_dim": OUT_DIM,
#         "note":   "model loads on first request (~20s cold start)",
#     }


# @app.post("/embed")
# async def embed(req: IdentityRequest, _=Depends(validate)):
#     return {
#         "identity":   req.identity,
#         "provider":   req.provider,
#         "z_identity": embed_identity(req.identity, req.provider),
#         "dim":        OUT_DIM,
#     }


# @app.post("/embed_batch")
# async def embed_batch(req: IdentityBatchRequest, _=Depends(validate)):
#     results = [
#         {
#             "identity":   r.identity,
#             "provider":   r.provider,
#             "z_identity": embed_identity(r.identity, r.provider),
#             "dim":        OUT_DIM,
#         }
#         for r in req.identities
#     ]
#     return {"embeddings": results, "total": len(results)}




"""
Stage 4 — Cross-Cloud Identity Linking
Platform: Modal GPU T4 (16GB memory)
Model: google/flan-t5-base + contrastive learning projection head
Output: 128-dim normalized identity embeddings
"""

import modal
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import hashlib
import secrets as _secrets

# ============================================================
# MODAL IMAGE
# ============================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.43.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
    )
)

app = modal.App("stage4-identity-embedding", image=image)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_BASE = "google/flan-t5-base"
OUT_DIM = 128
HIDDEN_DIM = 512


# ============================================================
# MODEL DEFINITION
# ============================================================
class IdentityEmbedder(nn.Module):
    """Flan-T5 encoder + projection head for identity embeddings"""
    
    def __init__(self):
        super().__init__()
        from transformers import T5EncoderModel
        
        # Load T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(MODEL_BASE)
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Projection head: 768 (T5-base hidden size) → 512 → 128
        self.projection = nn.Sequential(
            nn.Linear(768, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, OUT_DIM),
            nn.LayerNorm(OUT_DIM),
        )
    
    def forward(self, input_ids, attention_mask):
        # Get encoder output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        
        # Project to output dimension
        z_identity = self.projection(pooled)
        z_identity = F.normalize(z_identity, p=2, dim=-1)
        
        return z_identity


# ============================================================
# MODAL CLASS
# ============================================================
@app.cls(
    gpu="T4",
    memory=16384,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)
class IdentityEncoder:
    @modal.enter()
    def load_model(self):
        """Load model on container startup"""
        import torch
        from transformers import T5Tokenizer
        
        print(f"🔄 Loading Stage 4 Identity Encoder...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")
        
        # Load tokenizer
        print(f"  Loading tokenizer from {MODEL_BASE}...")
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_BASE)
        
        # Load model
        print(f"  Loading model...")
        self.model = IdentityEmbedder()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Stage 4 ready — output dim: {OUT_DIM}")
    
    def _embed(self, identity: str, provider: str, entity_type: str = "User") -> List[float]:
        """Generate embedding for a single identity"""
        # Create input text
        # Format: "Identity: {identity} from {provider} is a {entity_type}"
        text = f"Identity: {identity} from {provider} is a {entity_type}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=64,
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            z_identity = self.model(inputs["input_ids"], inputs["attention_mask"])
        
        return z_identity[0].cpu().tolist()
    
    @modal.method()
    async def embed(self, identity: str, provider: str, entity_type: str = "User") -> dict:
        """Generate 128-dim embedding for an identity"""
        z = self._embed(identity, provider, entity_type)
        
        return {
            "identity": identity,
            "provider": provider,
            "z_identity": z,
            "dim": OUT_DIM,
        }
    
    @modal.method()
    async def embed_batch(self, identities: List[Dict[str, str]]) -> List[dict]:
        """Batch embedding for multiple identities"""
        results = []
        for item in identities:
            z = self._embed(
                item.get("identity", ""),
                item.get("provider", "AWS"),
                item.get("entity_type", "User")
            )
            results.append({
                "identity": item.get("identity", ""),
                "provider": item.get("provider", "AWS"),
                "z_identity": z,
                "dim": OUT_DIM,
            })
        return results


# ============================================================
# FASTAPI WRAPPER
# ============================================================
@app.function(
    gpu="T4",
    memory=16384,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import os
    import hashlib
    import secrets as _secrets
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel
    from typing import List, Optional
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_4_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    web = FastAPI(title="Stage 4 — Identity Embedding", version="1.0.0")
    
    encoder = IdentityEncoder()
    
    class EmbedRequest(BaseModel):
        identity: str
        provider: str
        entity_type: str = "User"
    
    class BatchEmbedRequest(BaseModel):
        identities: List[EmbedRequest]
    
    @web.get("/health")
    async def health():
        return {
            "stage": "4",
            "status": "ok",
            "model": MODEL_BASE,
            "out_dim": OUT_DIM,
        }
    
    @web.post("/embed")
    async def embed(req: EmbedRequest, _=Depends(validate)):
        result = await encoder.embed.remote.aio(req.identity, req.provider, req.entity_type)
        return result
    
    @web.post("/embed_batch")
    async def embed_batch(req: BatchEmbedRequest, _=Depends(validate)):
        identities = [{"identity": i.identity, "provider": i.provider, "entity_type": i.entity_type} 
                     for i in req.identities]
        results = await encoder.embed_batch.remote.aio(identities)
        return {"embeddings": results, "total": len(results)}
    
    return web