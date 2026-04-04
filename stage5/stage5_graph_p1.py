"""
STAGE 5 — GRAPH CONSTRUCTION & FEATURE FUSION
Phase 2 — Real embeddings from Stages 2, 3b, 4

Multi-Cloud Threat Detection Pipeline — Group 24

Reads:
  Stage 0a outputs:
    ./data/stage0/structured_events.parquet
    ./data/stage0/node_labels.parquet
    ./data/stage0/graph_snapshots/*.pkl

  HuggingFace embeddings (Phase 2):
    sohomn/stage2-log-embeddings     → z_log.parquet      (entity_id, z_log[256])
    sohomn/stage3b-cve-risk-and-embeddings → stage3b_embeddings.parquet
    sohomn/stage4-identity-embeddings → z_identity.parquet (identity@provider, z_identity[128])

Writes:
  ./data/stage5/graph.pt
  ./data/stage5/snapshots/snapshot_t*.pt
  ./data/stage5/graph_summary.json
  ./data/stage5/node_index.parquet
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
STAGE0_DIR  = Path("D:/DATA PIPELINE/Stage0a/data/stage0")
OUTPUT_DIR  = Path("./data/stage5")
T           = 20
SCENARIO_ID = "scenario_00000"

HF_TOKEN    = os.environ.get("HF_TOKEN", "")

# Embedding dimensions — must match Stages 2, 3b, 4 exactly
DIM_Z_LOG      = 256
DIM_Z_CVE      = 128
DIM_RISK       = 1
DIM_EXPLOIT    = 1
DIM_Z_IDENTITY = 128
DIM_TOTAL      = DIM_Z_LOG + DIM_Z_CVE + DIM_RISK + DIM_EXPLOIT + DIM_Z_IDENTITY  # 514

NODE_TYPES = ["User", "VM", "Container", "IP", "Role", "CVE", "CloudAccount"]
EDGE_TYPES = [
    "ASSUMES_ROLE", "ACCESS", "CONNECTS_TO",
    "EXPLOITS", "HAS_VULNERABILITY", "DEPLOYED_ON",
    "BELONGS_TO", "CROSS_CLOUD_ACCESS",
]

# Node types that have each embedding — used for zero-padding decisions
HAS_Z_LOG      = {"User", "VM", "Container", "IP", "Role", "CloudAccount"}
HAS_Z_CVE      = {"CVE"}
HAS_Z_IDENTITY = {"User"}


# ── EMBEDDING LOADER — PHASE 2 ────────────────────────────────────────────────
def load_embeddings():
    """
    Downloads real embedding parquets from HuggingFace and builds
    lookup dictionaries keyed by entity_id / cve_id / identity@provider.

    Returns:
        z_log_map:   {entity_id: np.array(256)}
        z_cve_map:   {cve_id: {"z_cve": array(128), "risk_score": float, "exploit_prob": float}}
        z_id_map:    {"identity@provider": np.array(128)}
    """
    from huggingface_hub import hf_hub_download

    print("Loading Phase 2 embeddings from HuggingFace...")

    # ── z_log from Stage 2 ───────────────────────────────────────────────────
    try:
        z_log_path = hf_hub_download(
            repo_id  = "sohomn/stage2-log-embeddings",
            filename = "z_log.parquet",
            token    = HF_TOKEN or None,
        )
        df_log   = pd.read_parquet(z_log_path)
        z_log_map = {
            str(row["entity_id"]): np.array(row["z_log"], dtype=np.float32)
            for _, row in df_log.iterrows()
        }
        print(f"  z_log loaded     : {len(z_log_map)} entities")
    except Exception as e:
        print(f"  z_log UNAVAILABLE: {e} — using zeros")
        z_log_map = {}

    # ── z_cve + risk scores from Stage 3b ────────────────────────────────────
    try:
        z_cve_path = hf_hub_download(
            repo_id  = "sohomn/stage3b-cve-risk-and-embeddings",
            filename = "stage3b_embeddings.parquet",
            token    = HF_TOKEN or None,
        )
        df_cve   = pd.read_parquet(z_cve_path)
        z_cve_map = {
            str(row["cve_id"]): {
                "z_cve":        np.array(row["z_cve"], dtype=np.float32),
                "risk_score":   float(row["risk_score"]),
                "exploit_prob": float(row["exploit_prob"]),
            }
            for _, row in df_cve.iterrows()
        }
        print(f"  z_cve loaded     : {len(z_cve_map)} CVEs")
    except Exception as e:
        print(f"  z_cve UNAVAILABLE: {e} — using zeros")
        z_cve_map = {}

    # ── z_identity from Stage 4 ───────────────────────────────────────────────
    try:
        z_id_path = hf_hub_download(
            repo_id  = "sohomn/stage4-identity-embeddings",
            filename = "z_identity.parquet",
            token    = HF_TOKEN or None,
        )
        df_id    = pd.read_parquet(z_id_path)
        z_id_map = {
            f"{row['identity']}@{row['provider']}": np.array(row["z_identity"], dtype=np.float32)
            for _, row in df_id.iterrows()
        }
        print(f"  z_identity loaded: {len(z_id_map)} identities")
    except Exception as e:
        print(f"  z_identity UNAVAILABLE: {e} — using zeros")
        z_id_map = {}

    return z_log_map, z_cve_map, z_id_map


def zero_pad(vec, dim: int) -> np.ndarray:
    """Return vec reshaped to dim, or zeros if vec is None."""
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    arr = np.array(vec, dtype=np.float32)
    if len(arr) >= dim:
        return arr[:dim]
    return np.concatenate([arr, np.zeros(dim - len(arr), dtype=np.float32)])


def build_node_feature_vector(
    node_id:   str,
    node_type: str,
    provider:  str,
    z_log_map: dict,
    z_cve_map: dict,
    z_id_map:  dict,
) -> np.ndarray:
    """
    Assembles the 514-dim feature vector:
    [z_log(256) | z_cve(128) | risk_score(1) | exploit_prob(1) | z_identity(128)]

    Node types that don't have a given embedding receive zero padding:
    - CVE nodes   → no z_log, no z_identity
    - IP nodes    → no z_identity, no z_cve
    - User nodes  → have all (z_log, z_identity; z_cve only if they are CVE nodes)
    """
    # z_log — available for all non-CVE node types
    if node_type in HAS_Z_LOG:
        z_log = zero_pad(z_log_map.get(node_id), DIM_Z_LOG)
    else:
        z_log = np.zeros(DIM_Z_LOG, dtype=np.float32)

    # z_cve — only for CVE nodes, keyed by CVE ID
    if node_type in HAS_Z_CVE:
        cve_data   = z_cve_map.get(node_id, {})
        z_cve      = zero_pad(cve_data.get("z_cve"),      DIM_Z_CVE)
        risk_score = np.array([cve_data.get("risk_score",   0.0)], dtype=np.float32)
        exploit_p  = np.array([cve_data.get("exploit_prob", 0.0)], dtype=np.float32)
    else:
        z_cve      = np.zeros(DIM_Z_CVE,  dtype=np.float32)
        risk_score = np.zeros(DIM_RISK,   dtype=np.float32)
        exploit_p  = np.zeros(DIM_EXPLOIT, dtype=np.float32)

    # z_identity — only for User nodes, keyed by "identity@provider"
    if node_type in HAS_Z_IDENTITY:
        id_key = f"{node_id}@{provider}"
        z_id   = zero_pad(z_id_map.get(id_key), DIM_Z_IDENTITY)
    else:
        z_id = np.zeros(DIM_Z_IDENTITY, dtype=np.float32)

    fv = np.concatenate([z_log, z_cve, risk_score, exploit_p, z_id])
    assert fv.shape == (DIM_TOTAL,), f"Feature vector wrong shape: {fv.shape}"
    return fv


# ── NODE REGISTRY ─────────────────────────────────────────────────────────────
class NodeRegistry:
    def __init__(self):
        self.type_to_nodes = defaultdict(list)
        self.node_to_idx   = {}
        self.node_to_provider = {}   # tracks provider per node for z_identity lookup

    def register(self, node_id: str, node_type: str, provider: str = "AWS"):
        key = (node_type, node_id)
        if key not in self.node_to_idx:
            idx = len(self.type_to_nodes[node_type])
            self.type_to_nodes[node_type].append(node_id)
            self.node_to_idx[key]         = idx
            self.node_to_provider[node_id] = provider

    def idx(self, node_id: str, node_type: str) -> int:
        return self.node_to_idx[(node_type, node_id)]

    def provider(self, node_id: str) -> str:
        return self.node_to_provider.get(node_id, "AWS")

    def count(self, node_type: str) -> int:
        return len(self.type_to_nodes[node_type])


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────
def build_graph(
    events_df:     pd.DataFrame,
    node_labels_df: pd.DataFrame,
    scenario_id:   str,
    z_log_map:     dict,
    z_cve_map:     dict,
    z_id_map:      dict,
) -> tuple:

    events = events_df[events_df.scenario_id == scenario_id].copy()
    labels = node_labels_df[node_labels_df.scenario_id == scenario_id].copy()

    # ── Step 1: Register all nodes ────────────────────────────────────────────
    registry = NodeRegistry()
    for _, row in events.iterrows():
        if row.entity_type in NODE_TYPES:
            registry.register(row.entity_id, row.entity_type,
                               provider=str(row.get("provider", "AWS")).split("_")[0])
        if row.target_type in NODE_TYPES and pd.notna(getattr(row, "target_id", None)):
            registry.register(row.target_id, row.target_type,
                               provider=str(row.get("provider", "AWS")).split("_")[0])

    print("  Nodes registered:")
    for nt in NODE_TYPES:
        c = registry.count(nt)
        if c > 0:
            print(f"    {nt:15s}: {c}")

    # ── Step 2: Build node feature matrices ───────────────────────────────────
    data         = HeteroData()
    label_lookup = {row.node_id: int(row.compromised) for _, row in labels.iterrows()}

    phase = "phase2_real" if (z_log_map or z_cve_map or z_id_map) else "phase1_zeros"

    for node_type in NODE_TYPES:
        node_ids = registry.type_to_nodes[node_type]
        if not node_ids:
            continue

        features = np.stack([
            build_node_feature_vector(
                nid, node_type,
                registry.provider(nid),
                z_log_map, z_cve_map, z_id_map,
            )
            for nid in node_ids
        ])  # (num_nodes, 514)

        node_labels_arr = np.array(
            [label_lookup.get(nid, 0) for nid in node_ids], dtype=np.float32
        )

        data[node_type].x        = torch.tensor(features,        dtype=torch.float)
        data[node_type].y        = torch.tensor(node_labels_arr, dtype=torch.float)
        data[node_type].node_ids = node_ids

    # ── Step 3: Build edges ───────────────────────────────────────────────────
    edge_events = events[
        events.target_type.notna() &
        events.target_id.notna()   &
        events.entity_type.isin(NODE_TYPES) &
        events.target_type.isin(NODE_TYPES)
    ]

    edge_groups = defaultdict(lambda: {"src": [], "tgt": [], "malicious": []})
    for _, row in edge_events.iterrows():
        src_type = row.entity_type
        tgt_type = row.target_type
        relation = row.action
        if (src_type, row.entity_id) not in registry.node_to_idx:
            continue
        if (tgt_type, row.target_id) not in registry.node_to_idx:
            continue
        key = (src_type, relation, tgt_type)
        edge_groups[key]["src"].append(registry.idx(row.entity_id, src_type))
        edge_groups[key]["tgt"].append(registry.idx(row.target_id, tgt_type))
        edge_groups[key]["malicious"].append(int(row.malicious))

    print("\n  Edge types found:")
    for (src_t, rel, tgt_t), edges in edge_groups.items():
        pairs    = list(set(zip(edges["src"], edges["tgt"])))
        src_list = [p[0] for p in pairs]
        tgt_list = [p[1] for p in pairs]
        mal_lookup = {}
        for s, t, m in zip(edges["src"], edges["tgt"], edges["malicious"]):
            mal_lookup[(s, t)] = max(mal_lookup.get((s, t), 0), m)

        data[src_t, rel, tgt_t].edge_index = torch.tensor(
            [src_list, tgt_list], dtype=torch.long
        )
        data[src_t, rel, tgt_t].edge_label = torch.tensor(
            [mal_lookup[p] for p in pairs], dtype=torch.float
        )
        print(f"    ({src_t}, {rel}, {tgt_t}): {len(pairs)} edges")

    return data, registry, phase


# ── SNAPSHOT BUILDER ──────────────────────────────────────────────────────────
def build_snapshots(
    events_df:  pd.DataFrame,
    registry:   NodeRegistry,
    scenario_id: str,
    full_graph: HeteroData,
) -> list:
    events  = events_df[events_df.scenario_id == scenario_id].copy()
    events["window"] = events["t"].clip(0, T - 1)
    snapshots = []

    print(f"\n  Building {T} snapshots...")
    for window in range(T):
        window_events = events[events.window == window]
        snap = HeteroData()

        # Share node features with full graph
        for node_type in NODE_TYPES:
            if hasattr(full_graph[node_type], "x"):
                snap[node_type].x        = full_graph[node_type].x
                snap[node_type].y        = full_graph[node_type].y
                snap[node_type].node_ids = full_graph[node_type].node_ids

        # Only edges from this time window
        edge_events = window_events[
            window_events.target_type.notna() &
            window_events.target_id.notna()   &
            window_events.entity_type.isin(NODE_TYPES) &
            window_events.target_type.isin(NODE_TYPES)
        ]
        edge_groups = defaultdict(lambda: {"src": [], "tgt": []})
        for _, row in edge_events.iterrows():
            src_type = row.entity_type
            tgt_type = row.target_type
            if (src_type, row.entity_id) not in registry.node_to_idx:
                continue
            if (tgt_type, row.target_id) not in registry.node_to_idx:
                continue
            key = (src_type, row.action, tgt_type)
            edge_groups[key]["src"].append(registry.idx(row.entity_id, src_type))
            edge_groups[key]["tgt"].append(registry.idx(row.target_id, tgt_type))

        for (src_t, rel, tgt_t), edges in edge_groups.items():
            pairs = list(set(zip(edges["src"], edges["tgt"])))
            snap[src_t, rel, tgt_t].edge_index = torch.tensor(
                [[p[0] for p in pairs], [p[1] for p in pairs]], dtype=torch.long
            )

        snap["window"] = window
        snapshots.append(snap)

    active = sum(1 for s in snapshots if len(s.edge_types) > 0)
    print(f"  Active snapshots : {active}/{T}")
    return snapshots


# ── SUMMARY + NODE INDEX ──────────────────────────────────────────────────────
def write_summary(data, snapshots, registry, scenario_id, phase):
    summary = {
        "scenario_id":     scenario_id,
        "embedding_phase": phase,
        "dim_total":       DIM_TOTAL,
        "dim_breakdown":   {
            "z_log":       DIM_Z_LOG,
            "z_cve":       DIM_Z_CVE,
            "risk_score":  DIM_RISK,
            "exploit_prob":DIM_EXPLOIT,
            "z_identity":  DIM_Z_IDENTITY,
        },
        "nodes":           {nt: registry.count(nt) for nt in NODE_TYPES if registry.count(nt) > 0},
        "edge_types":      [f"{s}__{r}__{t}" for s, r, t in data.edge_types],
        "total_nodes":     sum(registry.count(nt) for nt in NODE_TYPES),
        "total_edge_types":len(data.edge_types),
        "num_snapshots":   len(snapshots),
        "snapshot_activity":[len(s.edge_types) for s in snapshots],
    }
    path = OUTPUT_DIR / "graph_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  graph_summary.json written")
    return summary


def write_node_index(registry):
    rows = [
        {"node_id": node_id, "node_type": node_type, "idx": idx}
        for (node_type, node_id), idx in registry.node_to_idx.items()
    ]
    df   = pd.DataFrame(rows)
    path = OUTPUT_DIR / "node_index.parquet"
    df.to_parquet(path, index=False)
    print(f"  node_index.parquet written → {len(df)} nodes")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STAGE 5 — GRAPH CONSTRUCTION (Phase 2: real embeddings)")
    print(f"  Scenario : {SCENARIO_ID}")
    print(f"  Node dim  : {DIM_TOTAL}")
    print(f"{'='*55}\n")

    # Load Stage 0a outputs
    print("Loading Stage 0a outputs...")
    events_df      = pd.read_parquet(STAGE0_DIR / "structured_events.parquet")
    node_labels_df = pd.read_parquet(STAGE0_DIR / "node_labels.parquet")
    print(f"  Events      : {len(events_df):,}")
    print(f"  Node labels : {len(node_labels_df):,}")

    # Load Phase 2 embeddings
    z_log_map, z_cve_map, z_id_map = load_embeddings()

    # Build graph
    print(f"\nBuilding full graph for {SCENARIO_ID}...")
    graph, registry, phase = build_graph(
        events_df, node_labels_df, SCENARIO_ID,
        z_log_map, z_cve_map, z_id_map,
    )

    # Build snapshots
    snapshots = build_snapshots(events_df, registry, SCENARIO_ID, graph)

    # Save outputs
    print("\nSaving outputs...")
    torch.save(graph, OUTPUT_DIR / "graph.pt")
    print("  graph.pt saved")

    for i, snap in enumerate(snapshots):
        torch.save(snap, OUTPUT_DIR / "snapshots" / f"snapshot_t{i:02d}.pt")
    print(f"  {len(snapshots)} snapshots saved")

    summary = write_summary(graph, snapshots, registry, SCENARIO_ID, phase)
    write_node_index(registry)

    print(f"\n{'='*55}")
    print(f"  COMPLETE")
    print(f"  Embedding phase  : {phase}")
    print(f"  Total nodes      : {summary['total_nodes']}")
    print(f"  Edge types       : {summary['total_edge_types']}")
    print(f"  z_log coverage   : {len(z_log_map)} entities")
    print(f"  z_cve coverage   : {len(z_cve_map)} CVEs")
    print(f"  z_identity       : {len(z_id_map)} identities")
    print(f"  Output           : {OUTPUT_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()