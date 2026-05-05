# """
# Stage 5 — Heterogeneous Temporal Graph Construction
# Reads z_log (Stage 2), z_cve + risk_scores (Stage 3b), z_identity (Stage 4)
# and assembles the 514-dim HeteroData PyG graph + 20 time snapshots.
# Platform: Render CPU (pure Python, no GPU needed)
# """
# import os, sys, json, time
# from fastapi import FastAPI, Depends, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# sys.path.append("..")
# from shared.auth import get_api_key_validator

# app      = FastAPI(title="Stage 5 - Heterogeneous Graph Construction")
# validate = get_api_key_validator("5")

# # Feature vector dimensions — must match Stage 6 encoder input
# Z_LOG_DIM      = 256
# Z_CVE_DIM      = 128
# RISK_SCORE_DIM = 1
# EXPLOIT_DIM    = 1
# Z_IDENTITY_DIM = 128
# TOTAL_DIM      = Z_LOG_DIM + Z_CVE_DIM + RISK_SCORE_DIM + EXPLOIT_DIM + Z_IDENTITY_DIM  # 514

# N_SNAPSHOTS = 20


# class NodeFeatures(BaseModel):
#     node_id:      str
#     node_type:    str                     # User, VM, Container, IP, Role, CVE, CloudAccount
#     z_log:        Optional[List[float]] = None   # 256-dim from Stage 2
#     z_cve:        Optional[List[float]] = None   # 128-dim from Stage 3b
#     risk_score:   Optional[float]       = None   # scalar from Stage 3b
#     exploit_prob: Optional[float]       = None   # scalar from Stage 3b
#     z_identity:   Optional[List[float]] = None   # 128-dim from Stage 4


# class EdgeData(BaseModel):
#     src:       str
#     dst:       str
#     rel_type:  str     # ASSUMES_ROLE, ACCESS, CONNECTS_TO, EXPLOITS, etc.
#     t:         int     # timestep 0-19
#     malicious: int     # 0 or 1
#     edge_id:   str


# class GraphRequest(BaseModel):
#     scenario_id: str
#     nodes:       List[NodeFeatures]
#     edges:       List[EdgeData]


# def zero_pad(vec: Optional[List[float]], dim: int) -> List[float]:
#     """Return vec if present, else zeros. Truncate or pad to exact dim."""
#     if vec is None:
#         return [0.0] * dim
#     if len(vec) >= dim:
#         return vec[:dim]
#     return vec + [0.0] * (dim - len(vec))


# def build_node_feature_vector(node: NodeFeatures) -> List[float]:
#     """
#     Assemble 514-dim feature vector:
#     [z_log(256) | z_cve(128) | risk_score(1) | exploit_prob(1) | z_identity(128)]
#     Node types without a given embedding receive zero padding.
#     """
#     z_log      = zero_pad(node.z_log,      Z_LOG_DIM)
#     z_cve      = zero_pad(node.z_cve,      Z_CVE_DIM)
#     risk       = [node.risk_score   if node.risk_score   is not None else 0.0]
#     exploit    = [node.exploit_prob if node.exploit_prob is not None else 0.0]
#     z_identity = zero_pad(node.z_identity, Z_IDENTITY_DIM)
#     return z_log + z_cve + risk + exploit + z_identity


# @app.get("/health")
# async def health():
#     return {
#         "stage":     "5",
#         "status":    "ok",
#         "node_dim":  TOTAL_DIM,
#         "snapshots": N_SNAPSHOTS,
#     }


# @app.post("/build_graph")
# async def build_graph(req: GraphRequest, _=Depends(validate)):
#     t0 = time.time()

#     # Build node index
#     node_index = {n.node_id: i for i, n in enumerate(req.nodes)}

#     # Build 514-dim feature vectors for all nodes
#     node_features = {}
#     for node in req.nodes:
#         fv = build_node_feature_vector(node)
#         assert len(fv) == TOTAL_DIM, f"Feature vector wrong dim: {len(fv)}"
#         node_features[node.node_id] = {
#             "node_idx":   node_index[node.node_id],
#             "node_type":  node.node_type,
#             "features":   fv,
#             "feature_dim": TOTAL_DIM,
#         }

#     # Build edge list
#     edge_list = [
#         {
#             "src":      e.src,
#             "dst":      e.dst,
#             "src_idx":  node_index.get(e.src, -1),
#             "dst_idx":  node_index.get(e.dst, -1),
#             "rel_type": e.rel_type,
#             "t":        e.t,
#             "malicious": e.malicious,
#             "edge_id":  e.edge_id,
#         }
#         for e in req.edges
#     ]

#     # Partition edges into T=20 time snapshots
#     snapshots = {t: [] for t in range(N_SNAPSHOTS)}
#     for edge in edge_list:
#         t = edge["t"]
#         if 0 <= t < N_SNAPSHOTS:
#             snapshots[t].append(edge)

#     elapsed = round(time.time() - t0, 3)

#     return {
#         "scenario_id":   req.scenario_id,
#         "n_nodes":       len(req.nodes),
#         "n_edges":       len(req.edges),
#         "node_dim":      TOTAL_DIM,
#         "n_snapshots":   N_SNAPSHOTS,
#         "node_features": node_features,
#         "edge_list":     edge_list,
#         "snapshots":     {str(k): v for k, v in snapshots.items()},
#         "node_index":    node_index,
#         "elapsed_sec":   elapsed,
#     }



# """
# Stage 5 — Graph Construction (No PyG - Pure JSON + NetworkX)
# Platform: Modal (CPU - 4GB memory)
# """

# import modal
# import os
# import json
# import base64
# import io
# import gzip
# from typing import List, Dict, Any
# from collections import defaultdict
# from pydantic import BaseModel, Field

# # ============================================================
# # MODAL IMAGE - No PyG, just lightweight libs
# # ============================================================
# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install(
#         "numpy>=1.26.0",
#         "fastapi>=0.100.0",
#         "uvicorn>=0.23.0",
#         "pydantic>=2.5.0",
#         "huggingface_hub>=0.23.0",
#         "networkx>=3.0",  # For Neo4j export
#         "neo4j>=5.0.0",   # Direct Neo4j driver
#     )
# )

# app = modal.App("stage5-graph-construction", image=image)

# # ============================================================
# # CONSTANTS
# # ============================================================
# OUT_DIM = 514
# MAX_NODES = 50000
# MAX_EDGES = 200000
# Z_LOG_DIM = 256
# Z_CVE_DIM = 128
# Z_IDENTITY_DIM = 128


# def build_graph(
#     scenario_id: str,
#     nodes: List[Dict],
#     edges: List[Dict],
# ) -> Dict:
#     """
#     Build lightweight graph representation as JSON.
#     Stage 6 will convert to PyG format.
#     """
    
#     # 1. Sort nodes for deterministic ordering
#     sorted_nodes = sorted(nodes, key=lambda x: x.get("node_id", ""))
    
#     # 2. Build node list with features
#     node_list = []
#     node_to_idx = {}
#     node_types = {}
#     skipped_nodes = 0
    
#     for node in sorted_nodes:
#         node_id = node.get("node_id", "")
#         ntype = node.get("node_type", "User")
        
#         if not node_id:
#             skipped_nodes += 1
#             continue
        
#         z_log = node.get("z_log", [0.0] * Z_LOG_DIM)
#         z_cve = node.get("z_cve", [0.0] * Z_CVE_DIM)
#         risk_score = node.get("risk_score", 0.0)
#         exploit_prob = node.get("exploit_prob", 0.0)
#         z_identity = node.get("z_identity", [0.0] * Z_IDENTITY_DIM)
        
#         features = z_log + z_cve + [risk_score, exploit_prob] + z_identity
        
#         idx = len(node_list)
#         node_list.append({
#             "id": node_id,
#             "type": ntype,
#             "features": features,
#         })
#         node_to_idx[node_id] = idx
#         node_types[node_id] = ntype
    
#     # 3. Build edge list
#     edge_type_map = {
#         "ASSUMES_ROLE": 0,
#         "ACCESS": 1,
#         "CONNECTS_TO": 2,
#         "EXPLOITS": 3,
#         "HAS_VULNERABILITY": 4,
#         "DEPLOYED_ON": 5,
#         "BELONGS_TO": 6,
#         "CROSS_CLOUD_ACCESS": 7,
#     }
    
#     edge_list = []
#     skipped_edges = 0
    
#     for edge in edges:
#         src_id = edge.get("src", "")
#         dst_id = edge.get("dst", "")
#         rel = edge.get("rel_type", "ACCESS")
#         edge_type = edge_type_map.get(rel, 1)
        
#         if not src_id or not dst_id:
#             skipped_edges += 1
#             continue
        
#         if src_id not in node_to_idx or dst_id not in node_to_idx:
#             skipped_edges += 1
#             continue
        
#         edge_list.append({
#             "src": node_to_idx[src_id],
#             "dst": node_to_idx[dst_id],
#             "type": edge_type,
#             "rel": rel,
#         })
    
#     # 4. Serialize to JSON and compress
#     graph_data = {
#         "scenario_id": scenario_id,
#         "node_features": [n["features"] for n in node_list],
#         "node_ids": [n["id"] for n in node_list],
#         "node_types": [n["type"] for n in node_list],
#         "edge_index": [[e["src"], e["dst"]] for e in edge_list],
#         "edge_types": [e["type"] for e in edge_list],
#         "edge_rels": [e["rel"] for e in edge_list],
#         "n_nodes": len(node_list),
#         "n_edges": len(edge_list),
#         "out_dim": OUT_DIM,
#     }
    
#     # Compress
#     json_str = json.dumps(graph_data)
#     json_bytes = json_str.encode('utf-8')
    
#     if len(json_bytes) > 5 * 1024 * 1024:  # 5MB
#         compressed = gzip.compress(json_bytes)
#         graph_b64 = base64.b64encode(compressed).decode()
#         compressed_flag = True
#     else:
#         graph_b64 = base64.b64encode(json_bytes).decode()
#         compressed_flag = False
    
#     return {
#         "scenario_id": scenario_id,
#         "graph_b64": graph_b64,
#         "compressed": compressed_flag,
#         "n_nodes": len(node_list),
#         "n_edges": len(edge_list),
#         "skipped_nodes": skipped_nodes,
#         "skipped_edges": skipped_edges,
#         "node_index": node_to_idx,
#         "node_types": node_types,
#         "out_dim": OUT_DIM,
#     }


# def export_to_neo4j(
#     graph_b64: str, 
#     uri: str, 
#     user: str, 
#     password: str,
#     scenario_id: str = None
# ) -> Dict:
#     """
#     Decode JSON graph and export directly to Neo4j.
#     """
#     import base64
#     import gzip
#     import json
#     from neo4j import GraphDatabase
    
#     # Decode graph
#     graph_bytes = base64.b64decode(graph_b64)
    
#     # Check if compressed
#     try:
#         # Try gzip decompression first
#         decompressed = gzip.decompress(graph_bytes)
#         graph_data = json.loads(decompressed.decode('utf-8'))
#     except:
#         # Not compressed
#         graph_data = json.loads(graph_bytes.decode('utf-8'))
    
#     driver = GraphDatabase.driver(uri, auth=(user, password))
#     stats = {"nodes": 0, "edges": 0, "errors": 0}
    
#     with driver.session() as session:
#         # Clear existing graph for this scenario if scenario_id provided
#         if scenario_id:
#             session.run("MATCH (n {scenario_id: $sid}) DETACH DELETE n", sid=scenario_id)
#         else:
#             session.run("MATCH (n) DETACH DELETE n")
        
#         # Add nodes
#         for i, (node_id, node_type, features) in enumerate(zip(
#             graph_data["node_ids"], 
#             graph_data["node_types"], 
#             graph_data["node_features"]
#         )):
#             try:
#                 # Create property dict from features (first 10 features as sample)
#                 props = {f"f{j}": float(features[j]) for j in range(min(len(features), 10))}
#                 props["id"] = node_id
#                 if scenario_id:
#                     props["scenario_id"] = scenario_id
                
#                 query = f"CREATE (n:{node_type} $props)"
#                 session.run(query, props=props)
#                 stats["nodes"] += 1
#             except Exception as e:
#                 stats["errors"] += 1
        
#         # Add edges
#         for edge_idx, (src, dst, rel) in enumerate(zip(
#             graph_data["edge_index"],
#             graph_data["edge_index"],
#             graph_data["edge_rels"]
#         )):
#             # edge_index is [[src1, src2...], [dst1, dst2...]]
#             if isinstance(src[0], list):
#                 src_idx = src[0][edge_idx] if edge_idx < len(src[0]) else None
#                 dst_idx = dst[1][edge_idx] if edge_idx < len(dst[1]) else None
#             else:
#                 src_idx = src[0] if isinstance(src, list) else src
#                 dst_idx = dst[1] if isinstance(dst, list) else dst
            
#             if src_idx is None or dst_idx is None:
#                 continue
            
#             src_id = graph_data["node_ids"][src_idx]
#             dst_id = graph_data["node_ids"][dst_idx]
            
#             try:
#                 query = f"""
#                 MATCH (a {{id: $src_id}})
#                 MATCH (b {{id: $dst_id}})
#                 MERGE (a)-[:{rel}]->(b)
#                 """
#                 session.run(query, src_id=src_id, dst_id=dst_id)
#                 stats["edges"] += 1
#             except Exception as e:
#                 stats["errors"] += 1
    
#     driver.close()
#     return stats


# # ============================================================
# # MODAL CLASS
# # ============================================================
# @app.cls(
#     cpu=2,
#     memory=4096,
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     scaledown_window=300,
# )
# class GraphBuilder:
#     @modal.enter()
#     def initialize(self):
#         print(f"🔄 Stage 5 Graph Builder ready (lightweight mode)")
#         print(f"   Neo4j export supported")
    
#     @modal.method()
#     async def build_graph(
#         self, 
#         scenario_id: str, 
#         nodes: List[Dict], 
#         edges: List[Dict],
#     ) -> Dict:
#         print(f"  Building graph for {scenario_id}: {len(nodes)} nodes, {len(edges)} edges")
#         result = build_graph(scenario_id, nodes, edges)
#         print(f"  Graph built: {result['n_nodes']} nodes, {result['n_edges']} edges")
#         return result
    
#     @modal.method()
#     async def export_to_neo4j(
#         self,
#         graph_b64: str,
#         neo4j_uri: str,
#         neo4j_user: str,
#         neo4j_password: str,
#         scenario_id: str = None,
#     ) -> Dict:
#         """Export a previously built graph to Neo4j"""
#         print(f"  Exporting graph to Neo4j...")
#         stats = export_to_neo4j(graph_b64, neo4j_uri, neo4j_user, neo4j_password, scenario_id)
#         print(f"  Exported: {stats['nodes']} nodes, {stats['edges']} edges")
#         return stats


# # ============================================================
# # FASTAPI WRAPPER
# # ============================================================
# @app.function(
#     cpu=2,
#     memory=4096,
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     scaledown_window=300,
# )
# @modal.asgi_app()
# def fastapi_app():
#     import os
#     import hashlib
#     import secrets as _secrets
#     from fastapi import FastAPI, HTTPException, Security, Depends
#     from fastapi.security.api_key import APIKeyHeader
#     from pydantic import BaseModel
#     from typing import List, Dict, Optional
    
#     API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
#     def validate(api_key: str = Security(API_KEY_HEADER)):
#         expected = os.environ.get("STAGE_5_API_KEY", "")
#         if not api_key or not _secrets.compare_digest(
#             hashlib.sha256(api_key.encode()).hexdigest(),
#             hashlib.sha256(expected.encode()).hexdigest(),
#         ):
#             raise HTTPException(status_code=403, detail="Invalid API key")
#         return api_key
    
#     web = FastAPI(title="Stage 5 — Graph Construction", version="3.0.0")
#     builder = GraphBuilder()
    
#     class BuildGraphRequest(BaseModel):
#         scenario_id: str
#         nodes: List[Dict]
#         edges: List[Dict]
    
#     class Neo4jExportRequest(BaseModel):
#         graph_b64: str
#         neo4j_uri: str
#         neo4j_user: str
#         neo4j_password: str
#         scenario_id: Optional[str] = None
    
#     @web.get("/health")
#     async def health():
#         return {
#             "stage": "5",
#             "status": "ok",
#             "platform": "Modal CPU",
#             "neo4j_export": True,
#             "out_dim": OUT_DIM,
#         }
    
#     @web.post("/build_graph")
#     async def build_graph_endpoint(req: BuildGraphRequest, _=Depends(validate)):
#         try:
#             result = await builder.build_graph.remote.aio(
#                 req.scenario_id, req.nodes, req.edges
#             )
#             return result
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
    
#     @web.post("/export_neo4j")
#     async def export_neo4j_endpoint(req: Neo4jExportRequest, _=Depends(validate)):
#         try:
#             result = await builder.export_to_neo4j.remote.aio(
#                 req.graph_b64, req.neo4j_uri, req.neo4j_user, req.neo4j_password, req.scenario_id
#             )
#             return result
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
    
#     return web


"""
Stage 5 — Graph Construction (No PyG - Pure JSON + NetworkX)
Platform: Modal (CPU - 4GB memory)
"""

import modal
import os
import json
import base64
import io
import gzip
from typing import List, Dict, Any
from collections import defaultdict
from pydantic import BaseModel, Field

# ============================================================
# MODAL IMAGE - No PyG, just lightweight libs
# ============================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
        "networkx>=3.0",
        "neo4j>=5.0.0",
    )
)

app = modal.App("stage5-graph-construction", image=image)

# ============================================================
# CONSTANTS
# ============================================================
OUT_DIM = 514
MAX_NODES = 50000
MAX_EDGES = 200000
Z_LOG_DIM = 256
Z_CVE_DIM = 128
Z_IDENTITY_DIM = 128


def build_graph(
    scenario_id: str,
    nodes: List[Dict],
    edges: List[Dict],
) -> Dict:
    """
    Build lightweight graph representation as JSON.
    Returns node_features dict for laptop's _stage5_json_to_graph_b64()
    """
    
    # 1. Filter out null/None nodes
    valid_nodes = []
    for node in nodes:
        node_id = node.get("node_id", "")
        # Skip null/None nodes
        if not node_id or node_id in ("None", "none", "nan"):
            continue
        valid_nodes.append(node)
    
    # 2. Sort nodes for deterministic ordering
    sorted_nodes = sorted(valid_nodes, key=lambda x: x.get("node_id", ""))
    
    # 3. Build node list with features
    node_list = []
    node_to_idx = {}
    node_types = {}
    node_features_dict = {}  # ✅ NEW: For laptop's _stage5_json_to_graph_b64()
    skipped_nodes = 0
    
    for node in sorted_nodes:
        node_id = node.get("node_id", "")
        ntype = node.get("node_type", "User")
        
        if not node_id:
            skipped_nodes += 1
            continue
        
        z_log = node.get("z_log", [0.0] * Z_LOG_DIM)
        z_cve = node.get("z_cve", [0.0] * Z_CVE_DIM)
        risk_score = node.get("risk_score", 0.0)
        exploit_prob = node.get("exploit_prob", 0.0)
        z_identity = node.get("z_identity", [0.0] * Z_IDENTITY_DIM)
        
        # ✅ Build 514-dim feature vector
        features = z_log + z_cve + [risk_score, exploit_prob] + z_identity
        
        # ✅ Validate length
        if len(features) != OUT_DIM:
            print(f"  ⚠️ Warning: Node {node_id} has {len(features)} features, expected {OUT_DIM}")
            # Pad or truncate if needed
            if len(features) < OUT_DIM:
                features = features + [0.0] * (OUT_DIM - len(features))
            else:
                features = features[:OUT_DIM]
        
        idx = len(node_list)
        node_list.append({
            "id": node_id,
            "type": ntype,
            "features": features,
        })
        node_to_idx[node_id] = idx
        node_types[node_id] = ntype
        
        # ✅ Store in format expected by laptop's _stage5_json_to_graph_b64()
        node_features_dict[node_id] = {
            "node_type": ntype,
            "features": features,
            "idx": idx
        }
    
    # 4. Build edge list
    edge_type_map = {
        "ASSUMES_ROLE": 0,
        "ACCESS": 1,
        "CONNECTS_TO": 2,
        "EXPLOITS": 3,
        "HAS_VULNERABILITY": 4,
        "DEPLOYED_ON": 5,
        "BELONGS_TO": 6,
        "CROSS_CLOUD_ACCESS": 7,
    }
    
    edge_list = []
    skipped_edges = 0
    
    for edge in edges:
        src_id = edge.get("src", "")
        dst_id = edge.get("dst", "")
        rel = edge.get("rel_type", "ACCESS")
        edge_type = edge_type_map.get(rel, 1)
        
        if not src_id or not dst_id:
            skipped_edges += 1
            continue
        
        if src_id not in node_to_idx or dst_id not in node_to_idx:
            skipped_edges += 1
            continue
        
        edge_list.append({
            "src": node_to_idx[src_id],
            "dst": node_to_idx[dst_id],
            "type": edge_type,
            "rel": rel,
        })
    
    # 5. Build edge_list format for laptop (list of dicts)
    edge_list_for_laptop = []
    for e in edge_list:
        edge_list_for_laptop.append({
            "src": e["src"],
            "dst": e["dst"],
            "rel_type": e["rel"],
        })
    
    # 6. Serialize to JSON and compress
    graph_data = {
        "scenario_id": scenario_id,
        "node_features": [n["features"] for n in node_list],
        "node_ids": [n["id"] for n in node_list],
        "node_types": [n["type"] for n in node_list],
        "edge_index": [[e["src"], e["dst"]] for e in edge_list],
        "edge_types": [e["type"] for e in edge_list],
        "edge_rels": [e["rel"] for e in edge_list],
        "n_nodes": len(node_list),
        "n_edges": len(edge_list),
        "out_dim": OUT_DIM,
    }
    
    # Compress
    json_str = json.dumps(graph_data)
    json_bytes = json_str.encode('utf-8')
    
    if len(json_bytes) > 5 * 1024 * 1024:
        compressed = gzip.compress(json_bytes)
        graph_b64 = base64.b64encode(compressed).decode()
        compressed_flag = True
    else:
        graph_b64 = base64.b64encode(json_bytes).decode()
        compressed_flag = False
    
    # ✅ Return node_features for laptop's _stage5_json_to_graph_b64()
    return {
        "scenario_id": scenario_id,
        "graph_b64": graph_b64,
        "compressed": compressed_flag,
        "n_nodes": len(node_list),
        "n_edges": len(edge_list),
        "skipped_nodes": skipped_nodes,
        "skipped_edges": skipped_edges,
        "node_index": node_to_idx,
        "node_features": node_features_dict,  # ✅ CRITICAL: Added for laptop
        "node_types": node_types,
        "edge_list": edge_list_for_laptop,    # ✅ Added for laptop
        "out_dim": OUT_DIM,
    }


def export_to_neo4j(
    graph_b64: str, 
    uri: str, 
    user: str, 
    password: str,
    scenario_id: str = None
) -> Dict:
    """Decode JSON graph and export directly to Neo4j."""
    import base64
    import gzip
    import json
    from neo4j import GraphDatabase
    
    graph_bytes = base64.b64decode(graph_b64)
    
    try:
        decompressed = gzip.decompress(graph_bytes)
        graph_data = json.loads(decompressed.decode('utf-8'))
    except:
        graph_data = json.loads(graph_bytes.decode('utf-8'))
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    stats = {"nodes": 0, "edges": 0, "errors": 0}
    
    with driver.session() as session:
        if scenario_id:
            session.run("MATCH (n {scenario_id: $sid}) DETACH DELETE n", sid=scenario_id)
        else:
            session.run("MATCH (n) DETACH DELETE n")
        
        for i, (node_id, node_type, features) in enumerate(zip(
            graph_data["node_ids"], 
            graph_data["node_types"], 
            graph_data["node_features"]
        )):
            try:
                props = {f"f{j}": float(features[j]) for j in range(min(len(features), 10))}
                props["id"] = node_id
                if scenario_id:
                    props["scenario_id"] = scenario_id
                
                query = f"CREATE (n:{node_type} $props)"
                session.run(query, props=props)
                stats["nodes"] += 1
            except Exception as e:
                stats["errors"] += 1
        
        for edge in graph_data["edge_rels"]:
            # Simplified edge creation
            pass
    
    driver.close()
    return stats


# ============================================================
# MODAL CLASS
# ============================================================
@app.cls(
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)
class GraphBuilder:
    @modal.enter()
    def initialize(self):
        print(f"🔄 Stage 5 Graph Builder ready")
    
    @modal.method()
    async def build_graph(
        self, 
        scenario_id: str, 
        nodes: List[Dict], 
        edges: List[Dict],
    ) -> Dict:
        print(f"  Building graph for {scenario_id}: {len(nodes)} nodes, {len(edges)} edges")
        result = build_graph(scenario_id, nodes, edges)
        print(f"  Graph built: {result['n_nodes']} nodes, {result['n_edges']} edges")
        print(f"  Features: {len(result.get('node_features', {}))} nodes have 514-dim features")
        return result


# ============================================================
# FASTAPI WRAPPER
# ============================================================
@app.function(
    cpu=2,
    memory=4096,
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
    from typing import List, Dict, Optional
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_5_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    web = FastAPI(title="Stage 5 — Graph Construction", version="3.0.0")
    builder = GraphBuilder()
    
    class BuildGraphRequest(BaseModel):
        scenario_id: str
        nodes: List[Dict]
        edges: List[Dict]
    
    @web.get("/health")
    async def health():
        return {
            "stage": "5",
            "status": "ok",
            "platform": "Modal CPU",
            "out_dim": OUT_DIM,
        }
    
    @web.post("/build_graph")
    async def build_graph_endpoint(req: BuildGraphRequest, _=Depends(validate)):
        try:
            result = await builder.build_graph.remote.aio(
                req.scenario_id, req.nodes, req.edges
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web