"""
Stage 6 + 7 — Parallel RGCN + Temporal GNN
Architecture:
  - Stage 6 (RGCN) and Stage 7 (GRU-GNN) run IN PARALLEL on the same graph
  - Stage 6 stays dormant (returns structural scores, waits)
  - Stage 7 runs sequentially through all 20 snapshots using Stage 6 embeddings
  - Only when Stage 7 completes does the combined result pass to Stage 8
  - Implemented as a single Modal app with asyncio.gather for true parallelism

Platform: Modal GPU T4
"""
import modal, os, torch, asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torch-geometric>=2.5.0",
        "transformers>=4.43.0",
        "huggingface_hub>=0.23.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
        "numpy>=1.26.0",
    )
)

app = modal.App("stage6-7-gnn-parallel", image=image)

NODE_DIM    = 514    # input feature dim from Stage 5
HIDDEN_DIM  = 256    # RGCN hidden dim
OUT_DIM     = 128    # final node embedding dim
N_SNAPSHOTS = 20
N_REL_TYPES = 10     # number of edge relation types


# ── Stage 6: RGCN Structural Threat Detection ─────────────────────────────────
@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class RGCNDetector:
    def __enter__(self):
        import torch.nn as nn

        class NodeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(NODE_DIM, HIDDEN_DIM),
                    nn.BatchNorm1d(HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                )
            def forward(self, x):
                return self.net(x)

        class PEFTAdapter(nn.Module):
            """Rank-16 bottleneck adapter — zero-init up projection."""
            def __init__(self, dim=HIDDEN_DIM, rank=16):
                super().__init__()
                self.down = nn.Linear(dim, rank, bias=False)
                self.up   = nn.Linear(rank, dim, bias=False)
                nn.init.zeros_(self.up.weight)   # starts as identity
            def forward(self, x):
                return x + self.up(torch.relu(self.down(x)))

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

        self.encoder     = NodeEncoder()
        self.adapter1    = PEFTAdapter()
        self.adapter2    = PEFTAdapter()
        self.threat_head = ThreatHead()

        # Relation-specific weight matrices — one per edge type
        self.W_r = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.01)
            for _ in range(N_REL_TYPES)
        ])
        self.W_0 = torch.nn.Parameter(torch.eye(HIDDEN_DIM))  # self-loop

        # DistMult relation vectors for edge anomaly scoring
        self.d_r = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(OUT_DIM))
            for _ in range(N_REL_TYPES)
        ])

        self.encoder.eval()
        self.threat_head.eval()
        print("✓ RGCN ready")

    @modal.method()
    def run(self, node_features: List[List[float]],
            edge_index: List[List[int]],
            edge_types: List[int]) -> dict:
        """
        Stage 6 forward pass.
        Returns: node embeddings (for Stage 7 GRU input) + threat scores.
        Stage 6 STAYS available — Stage 7 calls this for every snapshot.
        """
        X = torch.tensor(node_features, dtype=torch.float32)   # (N, 514)
        N = X.shape[0]

        # Step 1: per-node encoding
        h = self.encoder(X)                                      # (N, 256)
        h = self.adapter1(h)

        # Step 2: relation-aware message passing (simplified RGCN)
        if edge_index and len(edge_index[0]) > 0:
            src = torch.tensor(edge_index[0], dtype=torch.long)
            dst = torch.tensor(edge_index[1], dtype=torch.long)
            etypes = torch.tensor(edge_types, dtype=torch.long)

            agg = torch.zeros_like(h)
            for r in range(N_REL_TYPES):
                mask = (etypes == r)
                if mask.sum() == 0:
                    continue
                s_r = src[mask]
                d_r = dst[mask]
                msgs = h[s_r] @ self.W_r[r]           # (E_r, 256)
                agg.index_add_(0, d_r, msgs)

            # Self-loop
            h = torch.relu(agg + h @ self.W_0)

        h = self.adapter2(h)

        # Step 3: project to OUT_DIM
        proj = torch.nn.Linear(HIDDEN_DIM, OUT_DIM, bias=False)
        node_emb = torch.relu(proj(h))                           # (N, 128)

        # Step 4: threat scores
        threat_scores = self.threat_head(node_emb)               # (N, 1)

        # Step 5: edge anomaly scores via DistMult
        edge_anomaly = []
        if edge_index and len(edge_index[0]) > 0:
            src = torch.tensor(edge_index[0], dtype=torch.long)
            dst = torch.tensor(edge_index[1], dtype=torch.long)
            etypes = torch.tensor(edge_types, dtype=torch.long)
            for i in range(len(src)):
                r = min(etypes[i].item(), N_REL_TYPES - 1)
                score = torch.sigmoid(
                    (node_emb[src[i]] * self.d_r[r] * node_emb[dst[i]]).sum()
                ).item()
                edge_anomaly.append(score)

        return {
            "node_embeddings": node_emb.tolist(),    # (N, 128) — fed to Stage 7 GRU
            "threat_scores":   threat_scores.squeeze(-1).tolist(),  # (N,)
            "edge_anomaly":    edge_anomaly,
            "n_nodes":         N,
            "embedding_dim":   OUT_DIM,
        }


# ── Stage 7: Temporal GNN with GRU ───────────────────────────────────────────
@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class TemporalGNN:
    def __enter__(self):
        import torch.nn as nn

        class GRUHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru        = nn.GRU(OUT_DIM, OUT_DIM, batch_first=True)
                self.threat_head = nn.Linear(OUT_DIM, 1)
            def forward(self, h_seq):
                # h_seq: (N, T, 128)
                out, _ = self.gru(h_seq)          # (N, T, 128)
                preds  = torch.sigmoid(
                    self.threat_head(out)          # (N, T, 1)
                ).squeeze(-1)                      # (N, T)
                return out, preds

        self.gru_head = GRUHead()
        self.gru_head.eval()
        print("✓ Temporal GNN ready")

    @modal.method()
    def run(self,
            snapshot_embeddings: List[List[List[float]]],  # (T, N, 128)
            ) -> dict:
        """
        Stage 7 forward pass.
        snapshot_embeddings: list of T node embedding matrices from Stage 6.
        Each matrix is (N, 128) — one per time snapshot.
        Returns next-step compromise predictions for all nodes at all timesteps.
        """
        T = len(snapshot_embeddings)
        N = len(snapshot_embeddings[0]) if T > 0 else 0

        # Stack to (N, T, 128)
        h_seq = torch.tensor(
            [[snapshot_embeddings[t][n] for t in range(T)] for n in range(N)],
            dtype=torch.float32
        )

        with torch.no_grad():
            gru_out, preds = self.gru_head(h_seq)   # preds: (N, T)

        # Final hidden state = last timestep GRU output per node
        final_hidden = gru_out[:, -1, :].tolist()   # (N, 128)

        return {
            "next_step_predictions": preds.tolist(),    # (N, T) — y_hat(t+1)
            "final_hidden":          final_hidden,      # (N, 128) — last GRU state
            "n_nodes":               N,
            "n_timesteps":           T,
        }


# ── FastAPI wrapper — parallel orchestration of Stage 6 + 7 ──────────────────
@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import os, secrets, hashlib, asyncio
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel
    from typing import List

    web            = FastAPI(title="Stage 6+7 - Parallel RGCN + Temporal GNN")
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_67_API_KEY", "")
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    rgcn     = RGCNDetector()
    temporal = TemporalGNN()

    class SnapshotData(BaseModel):
        t:           int
        node_features: List[List[float]]   # (N, 514)
        edge_index:  List[List[int]]       # [[src...], [dst...]]
        edge_types:  List[int]             # one per edge

    class ParallelRequest(BaseModel):
        scenario_id: str
        snapshots:   List[SnapshotData]   # T=20 snapshots

    @web.get("/health")
    async def health():
        return {
            "stage":  "6+7",
            "status": "ok",
            "note":   "Stage 6 RGCN and Stage 7 GRU run in parallel, merge before Stage 8",
        }

    @web.post("/predict")
    async def predict(req: ParallelRequest, _=Depends(validate)):
        """
        Parallel execution:
        1. Stage 6 (RGCN) runs on ALL snapshots simultaneously — structural scores
        2. Stage 7 (GRU) waits for all Stage 6 embeddings, then processes sequentially
        3. Both results merge here before returning to gateway → Stage 8
        """
        snapshots = sorted(req.snapshots, key=lambda s: s.t)

        # ── PARALLEL BLOCK ────────────────────────────────────────────────────
        # Stage 6 runs on each snapshot independently — can all run in parallel
        # Stage 7 needs all Stage 6 embeddings first — starts after Stage 6 done
        # asyncio.gather runs Stage 6 calls concurrently

        async def run_stage6_snapshot(snap: SnapshotData) -> dict:
            return rgcn.run.remote(
                snap.node_features,
                snap.edge_index,
                snap.edge_types,
            )

        # Fire all Stage 6 calls in parallel
        stage6_results = await asyncio.gather(
            *[run_stage6_snapshot(s) for s in snapshots]
        )

        # ── Stage 6 dormant here — has returned all embeddings ────────────────
        # Collect final structural threat scores from Stage 6
        # Use the last snapshot's scores as the static structural assessment
        structural_scores = stage6_results[-1]["threat_scores"]   # (N,)
        edge_anomaly      = stage6_results[-1]["edge_anomaly"]

        # ── Stage 7 starts — needs all T snapshot embeddings ─────────────────
        # Stack snapshot embeddings: List[(N, 128)] → (T, N, 128)
        snapshot_embeddings = [r["node_embeddings"] for r in stage6_results]

        stage7_result = temporal.run.remote(snapshot_embeddings)

        # ── Merge Stage 6 + Stage 7 results → ready for Stage 8 ──────────────
        n_nodes = stage6_results[0]["n_nodes"]
        return {
            "scenario_id":        req.scenario_id,
            "n_nodes":            n_nodes,
            "n_snapshots":        len(snapshots),

            # Stage 6 outputs
            "S_structural":       structural_scores,          # (N,) threat scores
            "edge_anomaly":       edge_anomaly,               # per-edge anomaly

            # Stage 7 outputs
            "S_temporal":         stage7_result["next_step_predictions"],  # (N, T)
            "final_hidden":       stage7_result["final_hidden"],           # (N, 128)

            # Per-node final state for Stage 8
            "node_embeddings":    stage6_results[-1]["node_embeddings"],   # (N, 128)
        }

    return web