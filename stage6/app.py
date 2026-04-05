"""
Stage 6 — Heterogeneous RGCN Structural Threat Detection
Model: adarsh-aur/stage6-rgcn-security
Platform: Modal GPU T4
Input:  graph.pt (HeteroData from Stage 5) + node features
Output: node embeddings h_v (N, 128) + threat scores + edge anomaly scores
"""
import modal, os, torch
from typing import List, Dict, Optional
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0"
    )
    .pip_install(
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.18",
        "torch-geometric>=2.5.0",
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "numpy>=1.26.0",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
    )
)

app = modal.App("stage6-rgcn-detector", image=image)

REPO_ID    = "adarsh-aur/stage6-rgcn-security"
MODEL_FILE = "model_RGCN.pt"

# Architecture constants from model card
MAX_FDIM   = 1024
HIDDEN_DIM = 256
OUT_DIM    = 128
N_RELATIONS = 20
N_LAYERS    = 3
ADAPTER_RANK = 16


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class RGCNDetector:
    def __enter__(self):
        from huggingface_hub import hf_hub_download
        import torch.nn as nn
        from torch_geometric.nn import RGCNConv

        hf_token = os.environ.get("HF_TOKEN", "")

        # ── Model Architecture (matches training code) ────────────────────────
        class PEFTAdapter(nn.Module):
            def __init__(self, dim=HIDDEN_DIM, rank=ADAPTER_RANK):
                super().__init__()
                self.down = nn.Linear(dim, rank, bias=False)
                self.up   = nn.Linear(rank, dim, bias=False)
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
                # Auto-pad or truncate to MAX_FDIM
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
                self.encoder     = NodeEncoder()
                self.convs       = nn.ModuleList([
                    RGCNConv(HIDDEN_DIM, HIDDEN_DIM if i < N_LAYERS-1 else OUT_DIM,
                             num_relations=N_RELATIONS)
                    for i in range(N_LAYERS)
                ])
                self.adapters    = nn.ModuleList([
                    PEFTAdapter(HIDDEN_DIM if i < N_LAYERS-1 else OUT_DIM)
                    for i in range(N_LAYERS)
                ])
                self.threat_head = ThreatHead()
                # DistMult relation vectors for edge anomaly scoring
                self.d_r = nn.ParameterList([
                    nn.Parameter(torch.ones(OUT_DIM))
                    for _ in range(N_RELATIONS)
                ])

            def forward(self, graph):
                """
                graph: PyG HeteroData from Stage 5
                Returns: h_v (N, 128), node_offsets dict, threat_logits
                """
                # Flatten HeteroData to homogeneous for RGCNConv
                node_types  = list(graph.node_types)
                node_feats  = []
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

                x_all = torch.cat(node_feats, dim=0)   # (N_total, feat_dim)
                h     = self.encoder(x_all)              # (N_total, HIDDEN_DIM)

                # Build flat edge_index and edge_type tensors
                edge_indices = []
                edge_types   = []
                rel_idx      = 0
                edge_type_map = {}

                for edge_type in graph.edge_types:
                    src_type, rel, dst_type = edge_type
                    if src_type not in node_offsets or dst_type not in node_offsets:
                        continue
                    if not hasattr(graph[edge_type], 'edge_index'):
                        continue
                    ei         = graph[edge_type].edge_index.clone()
                    src_offset = node_offsets[src_type][0]
                    dst_offset = node_offsets[dst_type][0]
                    ei[0]     += src_offset
                    ei[1]     += dst_offset
                    edge_indices.append(ei)

                    if rel not in edge_type_map:
                        edge_type_map[rel] = rel_idx % N_RELATIONS
                        rel_idx += 1
                    edge_types.append(
                        torch.full((ei.shape[1],), edge_type_map[rel], dtype=torch.long)
                    )

                if edge_indices:
                    flat_ei    = torch.cat(edge_indices, dim=1)
                    flat_et    = torch.cat(edge_types,   dim=0)
                else:
                    flat_ei = torch.zeros(2, 0, dtype=torch.long)
                    flat_et = torch.zeros(0,    dtype=torch.long)

                # RGCN layers with PEFT adapters
                for conv, adapter in zip(self.convs, self.adapters):
                    h = torch.relu(conv(h, flat_ei, flat_et))
                    h = adapter(h)   # (N_total, dim)

                # h is now (N_total, OUT_DIM=128)
                threat_logits = self.threat_head(h)  # (N_total, 1)

                return h, node_offsets, threat_logits

        # ── Load trained weights ──────────────────────────────────────────────
        print(f"Loading {REPO_ID}/{MODEL_FILE}...")
        ckpt_path = hf_hub_download(
            repo_id  = REPO_ID,
            filename = MODEL_FILE,
            token    = hf_token or None,
        )
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = HeteroRGCN()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"✓ Stage 6 RGCN loaded — {sum(p.numel() for p in self.model.parameters()):,} params")

    @modal.method()
    def predict(self, graph_bytes: bytes) -> dict:
        """
        Takes serialised PyG HeteroData graph (torch.save bytes),
        returns node embeddings + threat scores for Stage 7.
        """
        import io
        graph = torch.load(io.BytesIO(graph_bytes), map_location="cpu", weights_only=False)

        with torch.no_grad():
            h_v, node_offsets, threat_logits = self.model(graph)

        threat_scores = threat_logits.squeeze(-1).tolist()

        # Edge anomaly scores via DistMult
        edge_anomaly = {}
        rel_idx = 0
        for edge_type in graph.edge_types:
            src_t, rel, dst_t = edge_type
            if src_t not in node_offsets or dst_t not in node_offsets:
                continue
            if not hasattr(graph[edge_type], "edge_index"):
                continue
            ei         = graph[edge_type].edge_index
            src_off    = node_offsets[src_t][0]
            dst_off    = node_offsets[dst_t][0]
            r          = rel_idx % len(self.model.d_r)
            scores     = []
            for i in range(ei.shape[1]):
                s = ei[0, i].item() + src_off
                d = ei[1, i].item() + dst_off
                if s < h_v.shape[0] and d < h_v.shape[0]:
                    score = torch.sigmoid(
                        (h_v[s] * self.model.d_r[r] * h_v[d]).sum()
                    ).item()
                    scores.append(score)
            if scores:
                edge_anomaly[f"{src_t}__{rel}__{dst_t}"] = scores
            rel_idx += 1

        return {
            "h_v":           h_v.tolist(),          # (N, 128) — fed to Stage 7
            "threat_scores": threat_scores,          # (N,)
            "edge_anomaly":  edge_anomaly,
            "n_nodes":       h_v.shape[0],
            "node_offsets":  {k: list(v) for k, v in node_offsets.items()},
        }


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import os, secrets, hashlib, base64
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel

    web            = FastAPI(title="Stage 6 - RGCN Structural Threat Detection")
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_6_API_KEY", "")
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    detector = RGCNDetector()

    class PredictRequest(BaseModel):
        # graph sent as base64-encoded torch.save bytes
        graph_b64:   str
        scenario_id: str

    @web.get("/health")
    async def health():
        return {
            "stage":  "6",
            "status": "ok",
            "model":  REPO_ID,
            "out_dim": OUT_DIM,
        }

    @web.post("/predict")
    async def predict(req: PredictRequest, _=Depends(validate)):
        graph_bytes = base64.b64decode(req.graph_b64)
        result      = detector.predict.remote(graph_bytes)
        result["scenario_id"] = req.scenario_id
        return result

    return web