"""
Stage 9+10 — Integrated Explainability & Report Generation
Model: Mistral-7B-Instruct-v0.2 + QLoRA (Stage 10) + BFS/GNNExplainer (Stage 9)
Platform: Modal GPU T4 (16GB)
"""
import modal
import os
import json
import re
import hashlib
import secrets as _secrets
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# MODAL IMAGE & APP
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "accelerate",
        "bitsandbytes",
        "peft",
        "huggingface_hub",
        "networkx",
        "numpy",
        "fastapi",
        "pydantic>=2.5.0",
    )
    .apt_install("build-essential")
)

app = modal.App("stage9-10-integrated", image=image)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REPO_ID = "sohomn/stage9-10-explain-report"
ADAPTER_FILE = "adapter_model.safetensors"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_THRESHOLD = 0.50

# Feature Slices for SHAP (514-dim vector)
FEATURE_SLICES = {
    "log": (0, 256),
    "cve": (256, 384),
    "risk": (384, 385),
    "exploit": (385, 386),
    "identity": (386, 514)
}

SEVERITY_TIERS = {"Critical": 0.90, "High": 0.75, "Medium": 0.50, "Low": 0.00}

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATED PREDICTOR (Stage 9 + Stage 10)
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="T4",
    memory=16384,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class IntegratedExplainReport:
    def __enter__(self):
        # 🔑 LAZY IMPORTS: Prevents local parse errors during `modal deploy`
        import torch
        import torch.nn as nn
        import networkx as nx
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.nx = nx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Stage 10 Model (Mistral-7B-QLoRA)
        self.tokenizer = None
        self.model = None
        self.model_available = False

        try:
            print("[STAGE10] Loading Mistral-7B-QLoRA...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL, 
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            # Try loading adapter
            try:
                adapter_path = hf_hub_download(repo_id=REPO_ID, filename=ADAPTER_FILE, token=os.getenv("HF_TOKEN"))
                self.model = PeftModel.from_pretrained(self.model, os.path.dirname(adapter_path))
                print("[STAGE10] QLoRA adapter loaded.")
            except Exception:
                print("[STAGE10] Adapter not found, using base model.")
            
            self.model.eval()
            self.model_available = True
        except Exception as e:
            print(f"[STAGE10] Model load failed: {e}")

    def _get_severity(self, score: float) -> str:
        for tier, thresh in SEVERITY_TIERS.items():
            if score >= thresh:
                return tier
        return "Low"

    def _extract_subgraph(self, nodes: List[Dict], edges: List[Dict], target_id: str) -> Dict:
        G = self.nx.DiGraph()
        for n in nodes:
            G.add_node(n["node_id"], type=n.get("node_type"))
        for e in edges:
            G.add_edge(e["src"], e["dst"], rel=e.get("rel_type"), score=e.get("anomaly_score", 0.0))

        if target_id not in G:
            return {"path": "Node not found", "subgraph_nodes": []}

        # k-hop BFS
        ancestors = self.nx.ancestors(G, target_id)
        try:
            path = []
            if ancestors:
                start = list(ancestors)[0] 
                try:
                    path = self.nx.shortest_path(G, start, target_id)
                except:
                    path = [target_id]
            else:
                path = [target_id]
            return {"path": " -> ".join(path), "subgraph_nodes": list(G.nodes())}
        except:
            return {"path": "Path extraction failed", "subgraph_nodes": []}

    def _compute_shap(self, features: List[float]) -> Dict[str, float]:
        phi = {}
        total_norm_sq = 0.0
        for name, (start, end) in FEATURE_SLICES.items():
            slice_vals = self.torch.tensor(features[start:end], dtype=self.torch.float32)
            norm = self.torch.norm(slice_vals).item()
            phi[f"phi_{name}"] = norm
            total_norm_sq += norm ** 2
        
        # L2 Normalization
        total_norm = total_norm_sq ** 0.5
        if total_norm > 1e-6:
            phi = {k: v / total_norm for k, v in phi.items()}
        
        primary = max(phi, key=phi.get)
        return phi, primary.replace("phi_", "")

    @modal.method()
    def explain_and_report(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        scenario_id: str,
        threshold: float = DEFAULT_THRESHOLD
    ) -> dict:
        flagged_nodes = [n for n in nodes if n.get("threat_score", 0) >= threshold]
        if not flagged_nodes:
            return {"reports": [], "scenario_id": scenario_id, "status": "No nodes flagged"}

        reports = []
        for node in flagged_nodes:
            # Stage 9: Explainability
            graph_info = self._extract_subgraph(nodes, edges, node["node_id"])
            shap_attribution, primary_driver = self._compute_shap(node.get("features", [0.0]*514))
            
            stage9_out = {
                "severity": self._get_severity(node["threat_score"]),
                "shap": shap_attribution,
                "attack_path": graph_info["path"],
                "subgraph_nodes": graph_info["subgraph_nodes"],
                "primary_driver": primary_driver,
                "remediation": f"Isolate {primary_driver} slice and review {node['node_id']}"
            }

            # Stage 10: Report Generation
            if self.model_available:
                prompt = f"""<s>[INST] Generate a security report for node {node['node_id']}.
Severity: {stage9_out['severity']}. Path: {stage9_out['attack_path']}. Driver: {primary_driver}.
Return JSON with keys: summary, attack_narrative, mitre_tactics, remediation_steps, estimated_blast_radius. [/INST]"""
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with self.torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=256, temperature=0.2)
                    text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    # Extract JSON
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    stage10_out = json.loads(match.group(0)) if match else {"error": "Invalid JSON"}
                except Exception as e:
                    stage10_out = {"summary": f"Generation failed: {str(e)}"}
            else:
                stage10_out = {
                    "summary": f"High risk detected for {node['node_id']}.",
                    "attack_narrative": "Manual review required.",
                    "mitre_tactics": ["TA0001"],
                    "remediation_steps": ["Investigate logs"],
                    "estimated_blast_radius": "Unknown"
                }

            reports.append({
                "node_id": node["node_id"],
                "stage9": stage9_out,
                "stage10": stage10_out
            })

        return {"reports": reports, "scenario_id": scenario_id, "status": "success"}

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
@app.function(gpu="T4", memory=16384, secrets=[modal.Secret.from_name("siem-pipeline-secrets")], container_idle_timeout=300)
@modal.asgi_app()
def fastapi_app():
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.getenv("STAGE_9_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    class Node(BaseModel):
        node_id: str
        node_type: str
        threat_score: float
        features: List[float]

    class Edge(BaseModel):
        src: str
        dst: str
        rel_type: str
        anomaly_score: float = 0.0

    class Request(BaseModel):
        nodes: List[Node]
        edges: List[Edge]
        scenario_id: str
        threshold: float = DEFAULT_THRESHOLD

    class Stage9Resp(BaseModel):
        severity: str
        shap: Dict[str, float]
        attack_path: str
        subgraph_nodes: List[str]
        primary_driver: str
        remediation: str

    class Stage10Resp(BaseModel):
        summary: str
        attack_narrative: str
        mitre_tactics: List[str]
        remediation_steps: List[str]
        estimated_blast_radius: str

    class Report(BaseModel):
        node_id: str
        stage9: Stage9Resp
        stage10: Stage10Resp

    class Response(BaseModel):
        reports: List[Report]
        scenario_id: str
        status: str

    web = FastAPI(title="Stage 9+10 Integrated")
    predictor = IntegratedExplainReport()

    @web.get("/health")
    async def health():
        return {
            "stage": "9+10",
            "status": "ok",
            "model": "Mistral-7B-QLoRA",
            "gpu": "T4",
            "loaded": predictor.model_available if hasattr(predictor, 'model_available') else False
        }

    @web.post("/stage9/explain_and_report", response_model=Response)
    async def predict(req: Request, _=Depends(validate)):
        res = predictor.explain_and_report.remote(
            nodes=[n.dict() for n in req.nodes],
            edges=[e.dict() for e in req.edges],
            scenario_id=req.scenario_id,
            threshold=req.threshold
        )
        return Response(**res)

    return web