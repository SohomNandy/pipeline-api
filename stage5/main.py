"""
Stage 5 — Heterogeneous Temporal Graph Construction
Reads z_log (Stage 2), z_cve + risk_scores (Stage 3b), z_identity (Stage 4)
and assembles the 514-dim HeteroData PyG graph + 20 time snapshots.
Platform: Render CPU (pure Python, no GPU needed)
"""
import os, sys, json, time
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 5 - Heterogeneous Graph Construction")
validate = get_api_key_validator("5")

# Feature vector dimensions — must match Stage 6 encoder input
Z_LOG_DIM      = 256
Z_CVE_DIM      = 128
RISK_SCORE_DIM = 1
EXPLOIT_DIM    = 1
Z_IDENTITY_DIM = 128
TOTAL_DIM      = Z_LOG_DIM + Z_CVE_DIM + RISK_SCORE_DIM + EXPLOIT_DIM + Z_IDENTITY_DIM  # 514

N_SNAPSHOTS = 20


class NodeFeatures(BaseModel):
    node_id:      str
    node_type:    str                     # User, VM, Container, IP, Role, CVE, CloudAccount
    z_log:        Optional[List[float]] = None   # 256-dim from Stage 2
    z_cve:        Optional[List[float]] = None   # 128-dim from Stage 3b
    risk_score:   Optional[float]       = None   # scalar from Stage 3b
    exploit_prob: Optional[float]       = None   # scalar from Stage 3b
    z_identity:   Optional[List[float]] = None   # 128-dim from Stage 4


class EdgeData(BaseModel):
    src:       str
    dst:       str
    rel_type:  str     # ASSUMES_ROLE, ACCESS, CONNECTS_TO, EXPLOITS, etc.
    t:         int     # timestep 0-19
    malicious: int     # 0 or 1
    edge_id:   str


class GraphRequest(BaseModel):
    scenario_id: str
    nodes:       List[NodeFeatures]
    edges:       List[EdgeData]


def zero_pad(vec: Optional[List[float]], dim: int) -> List[float]:
    """Return vec if present, else zeros. Truncate or pad to exact dim."""
    if vec is None:
        return [0.0] * dim
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def build_node_feature_vector(node: NodeFeatures) -> List[float]:
    """
    Assemble 514-dim feature vector:
    [z_log(256) | z_cve(128) | risk_score(1) | exploit_prob(1) | z_identity(128)]
    Node types without a given embedding receive zero padding.
    """
    z_log      = zero_pad(node.z_log,      Z_LOG_DIM)
    z_cve      = zero_pad(node.z_cve,      Z_CVE_DIM)
    risk       = [node.risk_score   if node.risk_score   is not None else 0.0]
    exploit    = [node.exploit_prob if node.exploit_prob is not None else 0.0]
    z_identity = zero_pad(node.z_identity, Z_IDENTITY_DIM)
    return z_log + z_cve + risk + exploit + z_identity


@app.get("/health")
async def health():
    return {
        "stage":     "5",
        "status":    "ok",
        "node_dim":  TOTAL_DIM,
        "snapshots": N_SNAPSHOTS,
    }


@app.post("/build_graph")
async def build_graph(req: GraphRequest, _=Depends(validate)):
    t0 = time.time()

    # Build node index
    node_index = {n.node_id: i for i, n in enumerate(req.nodes)}

    # Build 514-dim feature vectors for all nodes
    node_features = {}
    for node in req.nodes:
        fv = build_node_feature_vector(node)
        assert len(fv) == TOTAL_DIM, f"Feature vector wrong dim: {len(fv)}"
        node_features[node.node_id] = {
            "node_idx":   node_index[node.node_id],
            "node_type":  node.node_type,
            "features":   fv,
            "feature_dim": TOTAL_DIM,
        }

    # Build edge list
    edge_list = [
        {
            "src":      e.src,
            "dst":      e.dst,
            "src_idx":  node_index.get(e.src, -1),
            "dst_idx":  node_index.get(e.dst, -1),
            "rel_type": e.rel_type,
            "t":        e.t,
            "malicious": e.malicious,
            "edge_id":  e.edge_id,
        }
        for e in req.edges
    ]

    # Partition edges into T=20 time snapshots
    snapshots = {t: [] for t in range(N_SNAPSHOTS)}
    for edge in edge_list:
        t = edge["t"]
        if 0 <= t < N_SNAPSHOTS:
            snapshots[t].append(edge)

    elapsed = round(time.time() - t0, 3)

    return {
        "scenario_id":   req.scenario_id,
        "n_nodes":       len(req.nodes),
        "n_edges":       len(req.edges),
        "node_dim":      TOTAL_DIM,
        "n_snapshots":   N_SNAPSHOTS,
        "node_features": node_features,
        "edge_list":     edge_list,
        "snapshots":     {str(k): v for k, v in snapshots.items()},
        "node_index":    node_index,
        "elapsed_sec":   elapsed,
    }