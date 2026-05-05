"""
Stage 3b — CVE Risk Scoring
Platform: Modal GPU T4 (16GB memory)
Model: intfloat/e5-large-v2 (frozen) + MLP projection head + RiskHead
Training weights: sohomn/stage3b-cve-risk-and-embeddings/stage3b_best.pt
"""

import modal
import os
import torch
import torch.nn as nn
from typing import List, Dict
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
        "sentence-transformers>=2.7.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
    )
)

app = modal.App("stage3b-cve-risk", image=image)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_BASE = "intfloat/e5-large-v2"
HF_REPO = "sohomn/stage3b-cve-risk-and-embeddings"
WEIGHTS_FILE = "stage3b_best.pt"
OUT_DIM = 128
HIDDEN_DIM = 512
DROPOUT = 0.1


# ============================================================
# MODEL DEFINITION (Matches training notebook EXACTLY)
# ============================================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class E5Embedder(nn.Module):
    """Frozen e5 backbone + trainable projection → z_cve ∈ ℝ^128"""
    def __init__(self, backbone, in_dim=1024, out_dim=OUT_DIM):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(mean_pooling(out, attention_mask))


class RiskHead(nn.Module):
    """MLP regression head → risk_score ∈ [0,10], exploit_prob ∈ [0,1]"""
    def __init__(self, in_dim=OUT_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.risk_head = nn.Linear(32, 1)
        self.exploit_head = nn.Linear(32, 1)
    
    def forward(self, z_cve):
        s = self.shared(z_cve)
        return torch.sigmoid(self.risk_head(s)).squeeze(-1) * 10.0, torch.sigmoid(self.exploit_head(s)).squeeze(-1)


class Stage3bModel(nn.Module):
    """Full Stage 3b: e5 embedder + risk head"""
    def __init__(self, embedder, risk_head):
        super().__init__()
        self.embedder = embedder
        self.risk_head = risk_head
    
    def forward(self, input_ids, attention_mask):
        z_cve = self.embedder(input_ids, attention_mask)
        risk_score, exploit_prob = self.risk_head(z_cve)
        return z_cve, risk_score, exploit_prob


# ============================================================
# MODAL CLASS
# ============================================================
@app.cls(
    gpu="T4",
    memory=16384,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)

class CVERiskScorer:
    @modal.enter()
    def load_model(self):
        import traceback
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from huggingface_hub import hf_hub_download
            
            print(f"🔄 Loading Stage 3b CVE Risk Scorer on GPU...")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {self.device}")
            
            print(f"  Loading tokenizer from {MODEL_BASE}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
            
            print(f"  Loading backbone from {MODEL_BASE}...")
            backbone = AutoModel.from_pretrained(MODEL_BASE)
            backbone = backbone.to(self.device)
            
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
            
            print(f"  Creating embedder and risk head...")
            embedder = E5Embedder(backbone).to(self.device)
            risk_head = RiskHead().to(self.device)
            
            print(f"  Downloading weights from {HF_REPO}/{WEIGHTS_FILE}...")
            hf_token = os.environ.get("HF_TOKEN", "") or None
            weights_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=WEIGHTS_FILE,
                token=hf_token,
            )
            
            checkpoint = torch.load(weights_path, map_location="cpu")
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
            
            print(f"  Loading embedder weights...")
            embedder.load_state_dict(checkpoint["embedder"])
            
            print(f"  Loading risk_head weights...")
            risk_head.load_state_dict(checkpoint["risk_head"])
            
            self._model = Stage3bModel(embedder, risk_head)
            self._model.eval()
            self._model = self._model.to(self.device)
            
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print(f"  Total params: {total_params:,}")
            print(f"  Trainable params: {trainable_params:,}")
            
            print("  Running warmup inference...")
            dummy_text = ["query: vulnerability CVE-2024-1234 test for warmup"]
            dummy_inputs = self.tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
            with torch.no_grad():
                self._model(dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
            
            print(f"✅ Stage 3b ready")
        
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            traceback.print_exc()
            raise
    
    @modal.method()
    async def score(self, cves: List[Dict[str, str]]) -> List[Dict]:
        if not cves:
            return []
        
        if not hasattr(self, '_model') or self._model is None:
            raise RuntimeError("Model not loaded - check container logs")
        
        descriptions = [cve.get("description", "") for cve in cves]
        prefixed = [f"query: {desc}" for desc in descriptions]
        
        inputs = self.tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            z_cve, risk_scores, exploit_probs = self._model(inputs["input_ids"], inputs["attention_mask"])
        
        results = []
        for i, cve in enumerate(cves):
            results.append({
                "cve_id": cve.get("cve_id", ""),
                "z_cve": z_cve[i].cpu().tolist(),
                "risk_score": float(risk_scores[i].cpu().item()),
                "exploit_prob": float(exploit_probs[i].cpu().item()),
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
    from typing import List, Dict
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_3B_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    web = FastAPI(title="Stage 3b — CVE Risk Scoring", version="1.0.0")
    
    scorer = CVERiskScorer()
    
    class CVEScoreRequest(BaseModel):
        cves: List[Dict[str, str]]
    
    @web.get("/health")
    async def health():
        return {
            "stage": "3b",
            "status": "ok",
            "model": MODEL_BASE,
            "weights": f"{HF_REPO}/{WEIGHTS_FILE}",
            "out_dim": OUT_DIM,
        }
    
    @web.post("/score")
    async def score(req: CVEScoreRequest, _=Depends(validate)):
        if not req.cves:
            return {"results": []}
        
        results = await scorer.score.remote.aio(req.cves)
        return {"results": results}
    
    return web