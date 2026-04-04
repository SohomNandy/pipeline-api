"""
Stage 9 — Explainability and Risk Reporting
GNNExplainer: extracts minimal attack path subgraph per flagged node
SHAP: decomposes 514-dim feature vector into 5 semantic contributions
Platform: Render CPU (no GPU needed for explainability)
"""
import os, sys, time
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 9 - Explainability and Risk Reporting")
validate = get_api_key_validator("9")

# Feature vector slice boundaries — must match Stage 5
Z_LOG_SLICE      = (0,   256)
Z_CVE_SLICE      = (256, 384)
RISK_SLICE       = (384, 385)
EXPLOIT_SLICE    = (385, 386)
Z_IDENTITY_SLICE = (386, 514)

SEVERITY_TIERS = [
    (0.90, "Critical"),
    (0.75, "High"),
    (0.50, "Medium"),
    (0.00, "Low"),
]


class NodeData(BaseModel):
    node_id:       str
    node_type:     str
    threat_score:  float = Field(..., ge=0.0, le=1.0)
    features:      List[float]   # 514-dim


class EdgeData(BaseModel):
    src:      str
    dst:      str
    rel_type: str
    anomaly_score: Optional[float] = None


class ExplainRequest(BaseModel):
    scenario_id:    str
    nodes:          List[NodeData]
    edges:          List[EdgeData]
    threshold:      float = 0.50   # minimum threat_score to explain


def get_severity(score: float) -> str:
    for threshold, label in SEVERITY_TIERS:
        if score > threshold:
            return label
    return "Low"


def shap_decompose(features: List[float], threat_score: float) -> Dict:
    """
    Approximate SHAP decomposition by feature slice.
    Real SHAP requires the trained model — this is a proportional attribution stub
    that correctly attributes credit to each embedding component based on magnitude.
    Replace with shap.DeepExplainer(model, background) when model is trained.
    """
    import math

    slices = {
        "phi_log":      features[Z_LOG_SLICE[0]:Z_LOG_SLICE[1]],
        "phi_cve":      features[Z_CVE_SLICE[0]:Z_CVE_SLICE[1]],
        "phi_risk":     features[RISK_SLICE[0]:RISK_SLICE[1]],
        "phi_exploit":  features[EXPLOIT_SLICE[0]:EXPLOIT_SLICE[1]],
        "phi_identity": features[Z_IDENTITY_SLICE[0]:Z_IDENTITY_SLICE[1]],
    }

    # L2 norm of each slice as proxy for contribution
    norms = {k: math.sqrt(sum(x**2 for x in v)) for k, v in slices.items()}
    total = sum(norms.values()) or 1.0

    return {
        k: round((v / total) * threat_score, 4)
        for k, v in norms.items()
    }


def extract_attack_path(
    node_id: str,
    nodes: List[NodeData],
    edges: List[EdgeData],
    max_hops: int = 3,
) -> Dict:
    """
    GNNExplainer stub: extract minimal subgraph around a flagged node.
    Real GNNExplainer learns a soft mask over the computation graph.
    This returns the k-hop neighbourhood as the explanatory subgraph.
    """
    node_map   = {n.node_id: n for n in nodes}
    adj        = {}
    for e in edges:
        adj.setdefault(e.src, []).append(e)
        adj.setdefault(e.dst, []).append(e)

    # BFS up to max_hops
    visited_nodes = {node_id}
    visited_edges = []
    frontier      = [node_id]

    for _ in range(max_hops):
        next_frontier = []
        for nid in frontier:
            for edge in adj.get(nid, []):
                neighbour = edge.dst if edge.src == nid else edge.src
                visited_edges.append({
                    "src": edge.src, "dst": edge.dst,
                    "rel_type": edge.rel_type,
                    "anomaly_score": edge.anomaly_score,
                })
                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier

    subgraph_nodes = [
        {"node_id": nid, "node_type": node_map[nid].node_type,
         "threat_score": node_map[nid].threat_score}
        for nid in visited_nodes if nid in node_map
    ]

    # Build human-readable attack path description
    high_anomaly = sorted(
        [e for e in visited_edges if (e.get("anomaly_score") or 0) > 0.5],
        key=lambda x: x.get("anomaly_score", 0), reverse=True
    )[:5]

    path_str = " → ".join(
        f"{e['src']} --[{e['rel_type']}]--> {e['dst']}"
        for e in high_anomaly
    ) if high_anomaly else f"No high-anomaly edges in {max_hops}-hop neighbourhood"

    return {
        "subgraph_nodes": subgraph_nodes,
        "subgraph_edges": visited_edges[:20],
        "attack_path":    path_str,
        "n_hops":         max_hops,
    }


@app.get("/health")
async def health():
    return {
        "stage":  "9",
        "status": "ok",
        "note":   "GNNExplainer stub — replace shap_decompose with DeepExplainer when model trained",
    }


@app.post("/explain")
async def explain(req: ExplainRequest, _=Depends(validate)):
    t0      = time.time()
    reports = []

    flagged = [n for n in req.nodes if n.threat_score >= req.threshold]

    for node in flagged:
        severity    = get_severity(node.threat_score)
        shap_attrs  = shap_decompose(node.features, node.threat_score)
        attack_path = extract_attack_path(node.node_id, req.nodes, req.edges)

        # Determine primary driver of threat from SHAP
        primary_driver = max(shap_attrs, key=shap_attrs.get)
        driver_map = {
            "phi_log":      "anomalous log behaviour",
            "phi_cve":      "high CVE exposure",
            "phi_risk":     "elevated risk score",
            "phi_exploit":  "high exploit probability",
            "phi_identity": "suspicious cross-cloud identity",
        }

        reports.append({
            "node_id":        node.node_id,
            "node_type":      node.node_type,
            "threat_score":   node.threat_score,
            "severity":       severity,
            "shap":           shap_attrs,
            "primary_driver": driver_map.get(primary_driver, primary_driver),
            "attack_path":    attack_path,
            "remediation":    _get_remediation(severity, node.node_type),
            "scenario_id":    req.scenario_id,
        })

    return {
        "scenario_id":   req.scenario_id,
        "n_flagged":     len(flagged),
        "n_total_nodes": len(req.nodes),
        "threshold":     req.threshold,
        "reports":       reports,
        "elapsed_sec":   round(time.time() - t0, 3),
    }


def _get_remediation(severity: str, node_type: str) -> str:
    base = {
        "Critical": "Immediately isolate node, revoke all credentials, initiate IR playbook.",
        "High":     "Quarantine node within 1 hour, rotate credentials, alert SOC.",
        "Medium":   "Monitor closely, review access logs, schedule credential rotation.",
        "Low":      "Log for review, no immediate action required.",
    }.get(severity, "Review manually.")

    type_specific = {
        "User":         " Disable IAM user and review all recent AssumeRole events.",
        "VM":           " Snapshot VM for forensics before termination.",
        "CVE":          " Apply patch immediately or isolate affected systems.",
        "IP":           " Block IP at firewall/security group level.",
        "Role":         " Revoke role and audit all principals that assumed it.",
        "Container":    " Kill container and scan image for malicious layers.",
        "CloudAccount": " Enable CloudTrail/Activity Log and audit all cross-account access.",
    }.get(node_type, "")

    return base + type_specific