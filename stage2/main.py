# import modal
# import os
# import torch
# import sys
# from typing import List
# from pydantic import BaseModel

# # ============================================================
# # FIX 1: Better image with proper SentenceTransformer dependencies
# # ============================================================
# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("git", "gcc", "g++")  # Required for some sentence-transformers dependencies
#     .pip_install(
#         "sentence-transformers>=2.7.0",
#         "transformers>=4.43.0",
#         "torch>=2.0.0",
#         "huggingface_hub>=0.23.0",
#         "fastapi>=0.100.0",
#         "uvicorn>=0.23.0",
#         "pydantic>=2.5.0",
#         "numpy>=1.24.0",
#         "scikit-learn>=1.3.0",  # Often needed for sentence-transformers
#         "psutil>=5.9.0",  # Memory monitoring
#     )
# )

# app = modal.App("stage2-log-embeddings", image=image)

# MODEL_REPO = "Swapnanil09/bge-log-embeddings"
# IN_DIM = 1024
# OUT_DIM = 256

# # ============================================================
# # FIX 2: Add GPU support (BGE model runs faster on GPU)
# # ============================================================
# @app.cls(
#     gpu="T4",  # Adding GPU - will be faster and more stable
#     memory=4096,  # 4GB memory
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     container_idle_timeout=300,
#     allow_concurrent_inputs=20,  # Allow more concurrent requests
#     retries=2,
# )
# class LogEmbedder:
#     def __enter__(self):
#         import psutil
#         import gc
        
#         print(f"🚀 Starting Stage 2 Log Embedder...")
#         print(f"   Memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
#         # Force garbage collection
#         gc.collect()
        
#         from sentence_transformers import SentenceTransformer
#         import torch
        
#         # ============================================================
#         # FIX 3: Better model loading with fallback
#         # ============================================================
#         hf_token = os.environ.get("HF_TOKEN", "")
        
#         if not hf_token:
#             print("⚠️  WARNING: HF_TOKEN not set! May hit rate limits.")
        
#         print(f"📥 Loading model from {MODEL_REPO}...")
        
#         try:
#             # Try loading with token first
#             self.model = SentenceTransformer(
#                 MODEL_REPO,
#                 token=hf_token or None,
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#             )
#             print(f"   Device: {'GPU (T4)' if torch.cuda.is_available() else 'CPU'}")
#         except Exception as e:
#             print(f"❌ Failed to load from {MODEL_REPO}: {e}")
#             print("   Falling back to BAAI/bge-large-en-v1.5...")
            
#             # Fallback to base model
#             self.model = SentenceTransformer(
#                 "BAAI/bge-large-en-v1.5",
#                 token=hf_token or None,
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#             )
        
#         self.model.eval()
        
#         # ============================================================
#         # FIX 4: Projection head with proper device placement
#         # ============================================================
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.proj = torch.nn.Linear(IN_DIM, OUT_DIM, bias=False)
#         torch.nn.init.xavier_uniform_(self.proj.weight)
#         self.proj = self.proj.to(device)
#         self.proj.eval()
        
#         # Store device for later use
#         self.device = device
        
#         gpu_mem = 0
#         if torch.cuda.is_available():
#             gpu_mem = torch.cuda.memory_allocated() / 1024**2
        
#         print(f"✅ Model ready!")
#         print(f"   Embedding dim: {IN_DIM} → projected: {OUT_DIM}")
#         print(f"   GPU memory used: {gpu_mem:.1f} MB")
    
#     @modal.method()
#     def embed(self, entity_id: str, log_texts: List[str]) -> dict:
#         import torch
#         import gc
        
#         try:
#             # Cap at 20 log lines
#             texts = log_texts[:20]
            
#             if not texts:
#                 # Return zero embedding for entities with no logs
#                 return {
#                     "entity_id": entity_id,
#                     "z_log": [0.0] * OUT_DIM,
#                     "dim": OUT_DIM,
#                     "n_logs_used": 0,
#                     "model": MODEL_REPO,
#                     "warning": "No log texts provided",
#                 }
            
#             # ============================================================
#             # FIX 5: Better text preprocessing
#             # ============================================================
#             # BGE models expect 'query: ' or 'passage: ' prefix
#             # For log embedding, we use 'query: ' prefix
#             prefixed = []
#             for t in texts:
#                 if not t or len(t.strip()) == 0:
#                     prefixed.append("query: empty_log")
#                 else:
#                     # Truncate very long logs to prevent OOM
#                     if len(t) > 1024:
#                         t = t[:1024] + "..."
#                     prefixed.append(f"query: {t}")
            
#             # Encode all log lines
#             with torch.no_grad():
#                 raw_embs = self.model.encode(
#                     prefixed,
#                     normalize_embeddings=True,
#                     convert_to_tensor=True,
#                     show_progress_bar=False,
#                     batch_size=32,  # Smaller batch to prevent OOM
#                 )
                
#                 # raw_embs shape: (n_texts, 1024)
#                 # Mean pool across all log lines for this entity
#                 pooled = raw_embs.mean(dim=0)  # (1024,)
                
#                 # Normalize
#                 pooled = torch.nn.functional.normalize(
#                     pooled.unsqueeze(0), dim=-1
#                 )  # (1, 1024)
                
#                 # Project to 256-dim z_log
#                 z_log = self.proj(pooled)[0]  # (256,)
                
#                 # Ensure L2 normalized (Stage 5 expects normalized embeddings)
#                 z_log = torch.nn.functional.normalize(z_log.unsqueeze(0), dim=-1)[0]
            
#             # Clean up
#             del raw_embs
#             gc.collect()
            
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
            
#             return {
#                 "entity_id": entity_id,
#                 "z_log": z_log.cpu().tolist(),
#                 "dim": OUT_DIM,
#                 "n_logs_used": len(texts),
#                 "model": MODEL_REPO,
#             }
            
#         except torch.cuda.OutOfMemoryError as e:
#             print(f"❌ CUDA OOM for entity {entity_id}: {e}")
#             torch.cuda.empty_cache()
#             # Return zero embedding on OOM
#             return {
#                 "entity_id": entity_id,
#                 "z_log": [0.0] * OUT_DIM,
#                 "dim": OUT_DIM,
#                 "n_logs_used": 0,
#                 "model": MODEL_REPO,
#                 "error": "GPU out of memory",
#             }
#         except Exception as e:
#             print(f"❌ Embedding failed for {entity_id}: {e}")
#             # Return zero embedding on any error
#             return {
#                 "entity_id": entity_id,
#                 "z_log": [0.0] * OUT_DIM,
#                 "dim": OUT_DIM,
#                 "n_logs_used": 0,
#                 "model": MODEL_REPO,
#                 "error": str(e),
#             }


# # ============================================================
# # FIX 6: FastAPI with better error handling
# # ============================================================
# @app.function(
#     gpu="T4",
#     memory=4096,
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     container_idle_timeout=300,
#     allow_concurrent_inputs=20,
# )
# @modal.asgi_app()
# def fastapi_app():
#     import os
#     import secrets
#     import hashlib
#     import time
#     from fastapi import FastAPI, HTTPException, Security, Depends, Request
#     from fastapi.security.api_key import APIKeyHeader
#     from fastapi.responses import JSONResponse
#     from pydantic import BaseModel, Field
#     from contextlib import asynccontextmanager
    
#     # ============================================================
#     # FIX 7: Lifespan management
#     # ============================================================
#     @asynccontextmanager
#     async def lifespan(app: FastAPI):
#         # Startup
#         print("🔥 Stage 2 FastAPI starting up...")
#         app.state.start_time = time.time()
#         app.state.embedder = None
#         yield
#         # Shutdown
#         print("👋 Stage 2 FastAPI shutting down...")
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     web = FastAPI(
#         title="Stage 2 - Log Embeddings",
#         description="BGE-based log embedding for entity behavioural fingerprinting",
#         version="1.0.0",
#         lifespan=lifespan,
#     )
    
#     API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
#     def validate(api_key: str = Security(API_KEY_HEADER)):
#         expected = os.environ.get("STAGE_2_API_KEY", "")
        
#         # Skip validation if no key is set (development)
#         if not expected:
#             print("⚠️  WARNING: STAGE_2_API_KEY not set, skipping validation")
#             return api_key
        
#         if not api_key:
#             raise HTTPException(status_code=403, detail="Missing API key")
        
#         if not secrets.compare_digest(
#             hashlib.sha256(api_key.encode()).hexdigest(),
#             hashlib.sha256(expected.encode()).hexdigest(),
#         ):
#             raise HTTPException(status_code=403, detail="Invalid API key")
#         return api_key
    
#     # Lazy load embedder to avoid cold start timeout
#     embedder_instance = None
    
#     def get_embedder():
#         nonlocal embedder_instance
#         if embedder_instance is None:
#             embedder_instance = LogEmbedder()
#         return embedder_instance
    
#     class EmbedRequest(BaseModel):
#         entity_id: str = Field(..., description="Unique entity identifier")
#         log_texts: List[str] = Field(..., description="List of log lines for this entity", max_items=20)
    
#     class EmbedBatchRequest(BaseModel):
#         entities: List[EmbedRequest] = Field(..., description="List of entities to embed", max_items=100)
    
#     # ============================================================
#     # FIX 8: Enhanced health checks
#     # ============================================================
#     @web.get("/health")
#     async def health():
#         import torch
        
#         gpu_available = torch.cuda.is_available()
#         gpu_memory = 0
#         if gpu_available:
#             gpu_memory = torch.cuda.memory_allocated() / 1024**2
        
#         return {
#             "stage": "2",
#             "status": "ok",
#             "model": MODEL_REPO,
#             "in_dim": IN_DIM,
#             "out_dim": OUT_DIM,
#             "gpu_available": gpu_available,
#             "gpu_memory_mb": gpu_memory,
#             "uptime_seconds": time.time() - web.state.start_time,
#         }
    
#     @web.get("/ready")
#     async def ready():
#         """Kubernetes-style readiness probe"""
#         return {"status": "ready"}
    
#     @web.get("/live")
#     async def live():
#         """Kubernetes-style liveness probe"""
#         return {"status": "alive"}
    
#     # ============================================================
#     # FIX 9: Main endpoints with proper error handling
#     # ============================================================
#     @web.post("/embed")
#     async def embed(req: EmbedRequest, _=Depends(validate)):
#         try:
#             # Validate input
#             if not req.entity_id:
#                 raise HTTPException(status_code=400, detail="entity_id is required")
            
#             if len(req.log_texts) > 20:
#                 raise HTTPException(status_code=400, detail="max 20 log texts per entity")
            
#             embedder = get_embedder()
#             result = embedder.embed.remote(req.entity_id, req.log_texts)
#             return result
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             print(f"❌ Embed endpoint error: {e}")
#             raise HTTPException(status_code=500, detail=str(e))
    
#     @web.post("/embed_batch")
#     async def embed_batch(req: EmbedBatchRequest, _=Depends(validate)):
#         try:
#             if len(req.entities) > 100:
#                 raise HTTPException(status_code=400, detail="max 100 entities per batch")
            
#             embedder = get_embedder()
            
#             # Use asyncio.gather for parallel processing
#             import asyncio
#             tasks = [
#                 embedder.embed.remote(e.entity_id, e.log_texts)
#                 for e in req.entities
#             ]
#             results = await asyncio.gather(*tasks, return_exceptions=True)
            
#             # Handle exceptions in batch results
#             processed_results = []
#             for i, result in enumerate(results):
#                 if isinstance(result, Exception):
#                     processed_results.append({
#                         "entity_id": req.entities[i].entity_id,
#                         "error": str(result),
#                         "z_log": [0.0] * OUT_DIM,
#                     })
#                 else:
#                     processed_results.append(result)
            
#             return {
#                 "embeddings": processed_results,
#                 "total": len(processed_results),
#                 "successful": sum(1 for r in processed_results if "error" not in r),
#             }
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             print(f"❌ Batch embed error: {e}")
#             raise HTTPException(status_code=500, detail=str(e))
    
#     @web.exception_handler(Exception)
#     async def global_exception_handler(request: Request, exc: Exception):
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "error": str(exc),
#                 "entity_id": request.query_params.get("entity_id", "unknown"),
#             }
#         )
    
#     return web

"""
Stage 2 — Log Embedding
Platform: Modal GPU T4 (or CPU with 4GB+ memory)
Model: BAAI/bge-large-en-v1.5 fine-tuned with TripletLoss
Output: 256-dim L2-normalized embeddings (z_log)
"""

import modal
import os
import torch
from typing import List
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

app = modal.App("stage2-log-embeddings", image=image)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_REPO = "Swapnanil09/bge-log-embeddings"  # Your fine-tuned model
FALLBACK_MODEL = "BAAI/bge-large-en-v1.5"      # Fallback if fine-tuned fails
OUT_DIM = 256


# ============================================================
# MODAL CLASS
# ============================================================
@app.cls(
    gpu="T4",  # Use GPU for faster inference
    memory=8192,  # 8GB memory
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)
class LogEmbedder:
    @modal.enter()
    def load_model(self):
        """Load model on container startup"""
        import torch
        from sentence_transformers import SentenceTransformer
        
        print(f"🔄 Loading Stage 2 Log Embedder...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")
        
        # Try fine-tuned model first, fallback to base
        try:
            print(f"  Loading model from {MODEL_REPO}...")
            self.model = SentenceTransformer(
                MODEL_REPO,
                device=self.device,
            )
            print(f"  ✓ Loaded fine-tuned model")
        except Exception as e:
            print(f"  ⚠️ Failed to load fine-tuned model: {e}")
            print(f"  Falling back to {FALLBACK_MODEL}...")
            self.model = SentenceTransformer(
                FALLBACK_MODEL,
                device=self.device,
            )
            print(f"  ✓ Loaded fallback model")
        
        self.model.eval()
        
        # Projection head: 1024 → 256
        self.projection = torch.nn.Linear(1024, OUT_DIM, bias=False)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        self.projection = self.projection.to(self.device)
        self.projection.eval()
        
        print(f"✅ Stage 2 ready — output dim: {OUT_DIM}")
    
    @modal.method()
    async def embed(self, entity_id: str, log_texts: List[str]) -> dict:
        """Generate 256-dim embedding for an entity"""
        if not log_texts:
            return {
                "entity_id": entity_id,
                "z_log": [0.0] * OUT_DIM,
                "dim": OUT_DIM,
                "n_logs_used": 0,
                "warning": "No log texts provided",
            }
        
        # Cap at 20 log lines
        texts = log_texts[:20]
        
        # BGE models expect 'query: ' prefix
        prefixed = [f"query: {t}" for t in texts]
        
        with torch.no_grad():
            # Encode all log lines
            raw_embs = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            # Mean pool across all log lines
            pooled = raw_embs.mean(dim=0)
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled.unsqueeze(0), dim=-1)
            # Project to 256-dim
            z_log = self.projection(pooled)[0]
            # Final L2 normalize
            z_log = torch.nn.functional.normalize(z_log.unsqueeze(0), dim=-1)[0]
        
        return {
            "entity_id": entity_id,
            "z_log": z_log.cpu().tolist(),
            "dim": OUT_DIM,
            "n_logs_used": len(texts),
        }


# ============================================================
# FASTAPI WRAPPER
# ============================================================
@app.function(
    gpu="T4",
    memory=8192,
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
    from typing import List
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_2_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    web = FastAPI(title="Stage 2 — Log Embeddings", version="2.0.0")
    
    embedder = LogEmbedder()
    
    class EmbedRequest(BaseModel):
        entity_id: str
        log_texts: List[str]
    
    @web.get("/health")
    async def health():
        return {
            "stage": "2",
            "status": "ok",
            "model": MODEL_REPO,
            "out_dim": OUT_DIM,
        }
    
    @web.post("/embed")
    async def embed(req: EmbedRequest, _=Depends(validate)):
        if not req.log_texts:
            return {"results": []}
        
        result = await embedder.embed.remote.aio(req.entity_id, req.log_texts)
        return result
    
    return web