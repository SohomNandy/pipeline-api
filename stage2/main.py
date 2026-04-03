import modal, os, torch
from typing import List
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.7.0",
        "transformers>=4.43.0",
        "torch>=2.0.0",
        "huggingface_hub>=0.23.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
    )
)

app = modal.App("stage2-log-embeddings", image=image)

MODEL_REPO = "Swapnanil09/bge-log-embeddings"   # actual repo from model card usage
IN_DIM     = 1024                                # BGE-large output dim
OUT_DIM    = 256                                 # z_log dim expected by Stage 5


@app.cls(
    cpu=2,
    memory=3072,                  # 3GB — model is 1.2GB F32, need headroom
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class LogEmbedder:
    def __enter__(self):
        from sentence_transformers import SentenceTransformer

        print(f"Loading {MODEL_REPO}...")
        self.model = SentenceTransformer(
            MODEL_REPO,
            token=os.environ.get("HF_TOKEN", ""),
        )
        self.model.eval()

        # Projection head: 1024 → 256 (maps to z_log dim for Stage 5)
        self.proj = torch.nn.Linear(IN_DIM, OUT_DIM, bias=False)
        torch.nn.init.xavier_uniform_(self.proj.weight)
        self.proj.eval()

        print(f"✓ Model ready — embedding dim: {IN_DIM} → projected: {OUT_DIM}")

    @modal.method()
    def embed(self, entity_id: str, log_texts: List[str]) -> dict:
        # Cap at 20 log lines — same as original stub
        texts = log_texts[:20]

        # Prepend BGE query prefix as per model training convention
        prefixed = [f"query: {t}" for t in texts]

        # Encode all log lines — shape: (n_texts, 1024)
        with torch.no_grad():
            raw_embs = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            # Mean pool across all log lines for this entity
            pooled = raw_embs.mean(dim=0)                    # (1024,)
            pooled = torch.nn.functional.normalize(
                pooled.unsqueeze(0), dim=-1
            )                                                # (1, 1024)

            # Project to 256-dim z_log
            z_log = self.proj(pooled)[0]                     # (256,)

        return {
            "entity_id":    entity_id,
            "z_log":        z_log.tolist(),
            "dim":          OUT_DIM,
            "n_logs_used":  len(texts),
            "model":        MODEL_REPO,
        }


@app.function(
    cpu=2,
    memory=3072,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import os, secrets, hashlib
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader

    web            = FastAPI(title="Stage 2 - Log Embeddings")
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_2_API_KEY", "")
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    embedder = LogEmbedder()

    class EmbedRequest(BaseModel):
        entity_id: str
        log_texts: list[str]

    class EmbedBatchRequest(BaseModel):
        entities: list[EmbedRequest]

    @web.get("/health")
    async def health():
        return {
            "stage":  "2",
            "status": "ok",
            "model":  MODEL_REPO,
            "in_dim": IN_DIM,
            "out_dim": OUT_DIM,
        }

    @web.post("/embed")
    async def embed(req: EmbedRequest, _=Depends(validate)):
        result = embedder.embed.remote(req.entity_id, req.log_texts)
        return result

    @web.post("/embed_batch")
    async def embed_batch(req: EmbedBatchRequest, _=Depends(validate)):
        # Batch endpoint for Stage 5 — embed multiple entities in one call
        results = [
            embedder.embed.remote(e.entity_id, e.log_texts)
            for e in req.entities
        ]
        return {"embeddings": results, "total": len(results)}

    return web