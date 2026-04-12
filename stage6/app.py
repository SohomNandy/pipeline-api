"""
Stage 6 — Heterogeneous RGCN Structural Threat Detection
Model: adarsh-aur/stage6-rgcn-security
Platform: Modal GPU T4
"""
import modal
import os
import torch
import sys
from typing import List, Dict, Optional
from pydantic import BaseModel

# ============================================================
# FIX 1: Better image definition with proper dependency order
# ============================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "gcc", "g++", "make", "python3-dev")  # Required for compiling torch-scatter/sparse
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        index_url="https://download.pytorch.org/whl/cu118",  # CUDA 11.8 for T4
    )
    .pip_install(
        "torch-scatter",
        "torch-sparse",
        "torch-geometric>=2.5.0",
        extra_index_url="https://data.pyg.org/whl/torch-2.0.0+cu118.html",  # PYG precompiled wheels
    )
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "numpy>=1.26.0",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
        "psutil>=5.9.0",  # For memory monitoring
    )
)

app = modal.App("stage6-rgcn-detector", image=image)

REPO_ID = "adarsh-aur/stage6-rgcn-security"
MODEL_FILE = "model_RGCN.pt"

# Architecture constants
MAX_FDIM = 1024
HIDDEN_DIM = 256
OUT_DIM = 128
N_RELATIONS = 20
N_LAYERS = 3
ADAPTER_RANK = 16

# ============================================================
# FIX 2: Add memory monitoring and better error handling
# ============================================================
@app.cls(
    gpu="T4",
    memory=8192,  # 8GB memory for the container
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
    allow_concurrent_inputs=10,  # Allow concurrent requests
    retries=2,  # Retry on failure
)
class RGCNDetector:
    def __enter__(self):
        import psutil
        import gc
        
        print(f"🚀 Starting Stage 6 RGCN Detector...")
        print(f"   Memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        from huggingface_hub import hf_hub_download
        import torch.nn as nn
        from torch_geometric.nn import RGCNConv

        # ── Model Architecture ────────────────────────────────────────────
        class PEFTAdapter(nn.Module):
            def __init__(self, dim=HIDDEN_DIM, rank=ADAPTER_RANK):
                super().__init__()
                self.down = nn.Linear(dim, rank, bias=False)
                self.up = nn.Linear(rank, dim, bias=False)
                nn.init.zeros_(self.up.weight)
            def forward(self, x):
                return x + self.up(torch.relu(self.down(x)))

        class NodeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(MAX_FDIM, HIDDEN_DIM),
                    nn.BatchNorm1d(HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                )
            def forward(self, x):
                if x.shape[1] < MAX_FDIM:
                    x = torch.nn.functional.pad(x, (0, MAX_FDIM - x.shape[1]))
                elif x.shape[1] > MAX_FDIM:
                    x = x[:, :MAX_FDIM]
                return self.net(x)

        class ThreatHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(OUT_DIM, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
            def forward(self, x):
                return torch.sigmoid(self.net(x))

        class HeteroRGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = NodeEncoder()
                self.convs = nn.ModuleList([
                    RGCNConv(HIDDEN_DIM, HIDDEN_DIM if i < N_LAYERS-1 else OUT_DIM,
                             num_relations=N_RELATIONS)
                    for i in range(N_LAYERS)
                ])
                self.adapters = nn.ModuleList([
                    PEFTAdapter(HIDDEN_DIM if i < N_LAYERS-1 else OUT_DIM)
                    for i in range(N_LAYERS)
                ])
                self.threat_head = ThreatHead()
                self.d_r = nn.ParameterList([
                    nn.Parameter(torch.ones(OUT_DIM))
                    for _ in range(N_RELATIONS)
                ])

            def forward(self, graph):
                node_types = list(graph.node_types)
                node_feats = []
                node_offsets = {}
                offset = 0
                for nt in node_types:
                    if hasattr(graph[nt], 'x'):
                        x = graph[nt].x
                        node_offsets[nt] = (offset, offset + x.shape[0])
                        node_feats.append(x)
                        offset += x.shape[0]

                if not node_feats:
                    return torch.zeros(0, OUT_DIM), {}, torch.zeros(0, 1)

                x_all = torch.cat(node_feats, dim=0)
                h = self.encoder(x_all)

                edge_indices = []
                edge_types = []
                rel_idx = 0
                edge_type_map = {}

                for edge_type in graph.edge_types:
                    src_type, rel, dst_type = edge_type
                    if src_type not in node_offsets or dst_type not in node_offsets:
                        continue
                    if not hasattr(graph[edge_type], 'edge_index'):
                        continue
                    ei = graph[edge_type].edge_index.clone()
                    src_offset = node_offsets[src_type][0]
                    dst_offset = node_offsets[dst_type][0]
                    ei[0] += src_offset
                    ei[1] += dst_offset
                    edge_indices.append(ei)

                    if rel not in edge_type_map:
                        edge_type_map[rel] = rel_idx % N_RELATIONS
                        rel_idx += 1
                    edge_types.append(
                        torch.full((ei.shape[1],), edge_type_map[rel], dtype=torch.long)
                    )

                if edge_indices:
                    flat_ei = torch.cat(edge_indices, dim=1)
                    flat_et = torch.cat(edge_types, dim=0)
                else:
                    flat_ei = torch.zeros(2, 0, dtype=torch.long)
                    flat_et = torch.zeros(0, dtype=torch.long)

                for conv, adapter in zip(self.convs, self.adapters):
                    h = torch.relu(conv(h, flat_ei, flat_et))
                    h = adapter(h)

                threat_logits = self.threat_head(h)
                return h, node_offsets, threat_logits

        # ============================================================
        # FIX 3: Better model loading with fallbacks
        # ============================================================
        hf_token = os.environ.get("HF_TOKEN", "")
        
        if not hf_token:
            print("⚠️  WARNING: HF_TOKEN not set! Model loading may fail.")
        
        print(f"📥 Loading model from {REPO_ID}/{MODEL_FILE}...")
        
        try:
            ckpt_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILE,
                token=hf_token or None,
                local_dir="/tmp/model_cache",  # Cache locally
            )
            print(f"   Downloaded to: {ckpt_path}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise RuntimeError(f"Cannot load model: {e}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model = HeteroRGCN()
            
            # Handle different checkpoint formats
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
                
            self.model.eval()
            
            # Move to GPU
            self.model = self.model.cuda()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Total params: {total_params:,}")
            print(f"   Trainable params: {trainable_params:,}")
            print(f"   GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
            raise

    @modal.method()
    def predict(self, graph_bytes: bytes) -> dict:
        """
        Returns node embeddings + threat scores for Stage 7.
        """
        import io
        import psutil
        import gc
        
        # Memory check before processing
        mem_available = psutil.virtual_memory().available / 1024**3
        print(f"📊 Memory before predict: {mem_available:.1f} GB free")
        
        try:
            # Load graph
            graph = torch.load(io.BytesIO(graph_bytes), map_location="cpu", weights_only=False)
            
            # Move graph to GPU
            graph = graph.to("cuda")
            
            print(f"📊 Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges")
            
            with torch.no_grad():
                h_v, node_offsets, threat_logits = self.model(graph)
            
            # Move to CPU for serialization
            h_v_cpu = h_v.cpu()
            threat_scores = threat_logits.squeeze(-1).cpu().tolist()
            
            # Edge anomaly scores
            edge_anomaly = {}
            rel_idx = 0
            for edge_type in graph.edge_types:
                src_t, rel, dst_t = edge_type
                if src_t not in node_offsets or dst_t not in node_offsets:
                    continue
                if not hasattr(graph[edge_type], "edge_index"):
                    continue
                ei = graph[edge_type].edge_index
                src_off = node_offsets[src_t][0]
                dst_off = node_offsets[dst_t][0]
                r = rel_idx % len(self.model.d_r)
                scores = []
                for i in range(ei.shape[1]):
                    s = ei[0, i].item() + src_off
                    d = ei[1, i].item() + dst_off
                    if s < h_v_cpu.shape[0] and d < h_v_cpu.shape[0]:
                        score = torch.sigmoid(
                            (h_v_cpu[s] * self.model.d_r[r].cpu() * h_v_cpu[d]).sum()
                        ).item()
                        scores.append(score)
                if scores:
                    edge_anomaly[f"{src_t}__{rel}__{dst_t}"] = scores
                rel_idx += 1
            
            # Clean up GPU memory
            del graph
            gc.collect()
            torch.cuda.empty_cache()
            
            result = {
                "h_v": h_v_cpu.tolist(),
                "threat_scores": threat_scores,
                "edge_anomaly": edge_anomaly,
                "n_nodes": h_v_cpu.shape[0],
                "node_offsets": {k: list(v) for k, v in node_offsets.items()},
            }
            
            print(f"✅ Prediction complete. {len(threat_scores)} nodes processed.")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM: {e}")
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU out of memory. Try reducing graph size: {e}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

# ============================================================
# FIX 4: FastAPI with better error handling and health checks
# ============================================================
@app.function(
    gpu="T4",
    memory=8192,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    import os
    import secrets
    import hashlib
    import base64
    import time
    from fastapi import FastAPI, HTTPException, Security, Depends, Request
    from fastapi.security.api_key import APIKeyHeader
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from contextlib import asynccontextmanager
    
    # ============================================================
    # FIX 5: Startup/shutdown event handlers for better lifecycle
    # ============================================================
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("🔥 Stage 6 FastAPI starting up...")
        app.state.start_time = time.time()
        app.state.detector = None
        yield
        # Shutdown
        print("👋 Stage 6 FastAPI shutting down...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    web = FastAPI(
        title="Stage 6 - RGCN Structural Threat Detection",
        description="Heterogeneous RGCN for multi-cloud threat detection",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_6_API_KEY", "")
        
        # Skip validation if no key is set (development)
        if not expected:
            print("⚠️  WARNING: STAGE_6_API_KEY not set, skipping validation")
            return api_key
        
        if not api_key:
            raise HTTPException(status_code=403, detail="Missing API key")
        
        if not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    # Lazy load detector to avoid cold start issues
    detector_instance = None
    
    def get_detector():
        nonlocal detector_instance
        if detector_instance is None:
            detector_instance = RGCNDetector()
        return detector_instance
    
    class PredictRequest(BaseModel):
        graph_b64: str = Field(..., description="Base64-encoded torch.save bytes of graph.pt")
        scenario_id: str = Field(..., description="Scenario identifier for tracing")
    
    # ============================================================
    # FIX 6: Enhanced health check with model status
    # ============================================================
    @web.get("/health")
    async def health():
        gpu_available = torch.cuda.is_available()
        gpu_memory = 0
        if gpu_available:
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
        
        return {
            "stage": "6",
            "status": "ok",
            "model": REPO_ID,
            "out_dim": OUT_DIM,
            "gpu_available": gpu_available,
            "gpu_memory_mb": gpu_memory,
            "uptime_seconds": time.time() - web.state.start_time,
        }
    
    @web.get("/ready")
    async def ready():
        """Kubernetes-style readiness probe"""
        return {"status": "ready"}
    
    @web.get("/live")
    async def live():
        """Kubernetes-style liveness probe"""
        return {"status": "alive"}
    
    # ============================================================
    # FIX 7: Better error handling for predictions
    # ============================================================
    @web.post("/predict")
    async def predict(req: PredictRequest, request: Request, _=Depends(validate)):
        try:
            # Decode graph
            graph_bytes = base64.b64decode(req.graph_b64)
            
            # Validate size (prevent OOM)
            graph_size_mb = len(graph_bytes) / 1024 / 1024
            if graph_size_mb > 100:  # 100MB limit
                raise HTTPException(
                    status_code=413,
                    detail=f"Graph too large: {graph_size_mb:.1f} MB (max 100 MB)"
                )
            
            print(f"📥 Processing scenario {req.scenario_id} ({graph_size_mb:.1f} MB graph)")
            
            # Get detector and run prediction
            detector = get_detector()
            result = detector.predict.remote(graph_bytes)
            
            result["scenario_id"] = req.scenario_id
            result["status"] = "success"
            
            return result
            
        except base64.binascii.Error as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise HTTPException(status_code=507, detail=str(e))
            raise HTTPException(status_code=500, detail=f"Runtime error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    
    @web.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(exc),
                "scenario_id": request.query_params.get("scenario_id", "unknown"),
            }
        )
    
    return web