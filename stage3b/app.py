import modal, os, torch
from pydantic import BaseModel
from typing import List

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.43.0",
        "torch>=2.0.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
    )
)

app = modal.App("stage3b-cve-risk", image=image)

# ── Model class — loaded once per container, cached across requests ───────────
@app.cls(
    cpu=2,
    memory=2048,                  # 2GB — enough for e5-large-v2 (1.3GB)
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,   # keep warm 5 min
)
class CVERiskModel:
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModel

        MODEL_NAME = "intfloat/e5-large-v2"
        print(f"Loading {MODEL_NAME}...")

        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.mdl = AutoModel.from_pretrained(MODEL_NAME)
        self.mdl.eval()

        # Projection head: 1024 -> 128
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 128),
        )
        # Risk head: 128 -> 2 (risk_score, exploit_prob)
        self.risk_head = torch.nn.Linear(128, 2)

        # Load trained weights from HF if available
        try:
            from huggingface_hub import hf_hub_download
            import json

            weights_path = hf_hub_download(
                repo_id="sohomn/stage3b-cve-risk-and-embeddings",
                filename="pytorch_model.bin",
                token=os.environ.get("HF_TOKEN", ""),
            )
            state = torch.load(weights_path, map_location="cpu")
            # Load projection head weights if present
            proj_state = {k.replace("proj.", ""): v for k, v in state.items() if k.startswith("proj.")}
            risk_state = {k.replace("risk_head.", ""): v for k, v in state.items() if k.startswith("risk_head.")}
            if proj_state:
                self.proj.load_state_dict(proj_state)
                print("✓ Loaded trained projection head weights")
            if risk_state:
                self.risk_head.load_state_dict(risk_state)
                print("✓ Loaded trained risk head weights")
        except Exception as e:
            print(f"  Note: using untrained projection head ({e})")

        print("✓ Model ready")

    @modal.method()
    def score(self, cves: list) -> list:
        results = []
        for cve in cves:
            text   = f"query: {cve['description']}"
            inputs = self.tok(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            )
            with torch.no_grad():
                out  = self.mdl(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                emb  = torch.nn.functional.normalize(emb, dim=-1)
                z    = self.proj(emb)
                raw  = self.risk_head(z)
                risk = torch.sigmoid(raw[0][0]).item() * 10
                expl = torch.sigmoid(raw[0][1]).item()

            results.append({
                "cve_id":       cve["cve_id"],
                "z_cve":        z[0].tolist(),
                "risk_score":   round(risk, 4),
                "exploit_prob": round(expl, 4),
                "dim":          128,
            })
        return results


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
@app.function(
    cpu=2,
    memory=2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import os, secrets, hashlib
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader

    web            = FastAPI(title="Stage 3b - CVE Risk Scoring")
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_3B_API_KEY", "")
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    model = CVERiskModel()

    class CVERequest(BaseModel):
        cve_id:      str
        description: str

    class CVEBatchRequest(BaseModel):
        cves: List[CVERequest]

    @web.get("/health")
    async def health():
        return {
            "stage":  "3b",
            "status": "ok",
            "model":  "intfloat/e5-large-v2 + projection head",
            "repo":   "sohomn/stage3b-cve-risk-and-embeddings",
        }

    @web.post("/score")
    async def score(req: CVEBatchRequest, _=Depends(validate)):
        cves   = [c.dict() for c in req.cves]
        result = model.score.remote(cves)
        return {"results": result}

    return web