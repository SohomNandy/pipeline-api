# generate_training_data_no_shap.py
import pandas as pd
import json
import numpy as np
from pathlib import Path

DATA = Path("D:\DATA PIPELINE\pipeline api\stage8")

# Load Stage 8 results
stage8_df = pd.read_parquet(DATA / 'threat_scores.parquet')
print(f"Loaded {len(stage8_df)} nodes from Stage 8")

# Severity mapping
def get_severity(score):
    if score >= 0.90: return "Critical"
    if score >= 0.75: return "High"
    if score >= 0.50: return "Medium"
    return "Low"

# Generate synthetic SHAP based on node type and score
def generate_synthetic_shap(node_id: str, score: float, node_type: str = "User") -> dict:
    """Generate realistic SHAP values based on threat score and node type"""
    np.random.seed(hash(node_id) % 2**32)
    
    if node_type == "CVE":
        base = {"phi_log": 0.05, "phi_cve": 0.6, "phi_risk": 0.2, "phi_exploit": 0.1, "phi_identity": 0.05}
    elif node_type in ["VM", "Container"]:
        base = {"phi_log": 0.4, "phi_cve": 0.2, "phi_risk": 0.15, "phi_exploit": 0.15, "phi_identity": 0.1}
    else:  # User, IP, Role
        base = {"phi_log": 0.5, "phi_cve": 0.05, "phi_risk": 0.1, "phi_exploit": 0.05, "phi_identity": 0.3}
    
    # Scale by threat score
    scaled = {k: v * score * np.random.uniform(0.8, 1.2) for k, v in base.items()}
    # Normalize
    total = sum(scaled.values())
    if total > 0:
        scaled = {k: v / total for k, v in scaled.items()}
    
    return scaled

# Generate training pairs
training_pairs = []
system_prompt = "You are a cybersecurity analyst. Generate a concise threat report."

for idx, row in stage8_df.iterrows():
    node_id = row['node_id']
    score = float(row['final_threat_score'])
    severity = get_severity(score)
    node_type = row.get('node_type', 'User')
    
    # Generate synthetic SHAP
    shap = generate_synthetic_shap(node_id, score, node_type)
    
    input_data = {
        "node_id": node_id,
        "node_type": node_type,
        "threat_score": round(score, 4),
        "severity": severity,
        "shap_attribution": {k: round(v, 4) for k, v in shap.items()}
    }
    
    # Generate output based on severity and dominant SHAP
    dominant = max(shap, key=shap.get)
    
    if severity == "Critical":
        output = {
            "summary": f"🚨 CRITICAL: Node {node_id} ({node_type}) exhibits severe threat indicators with score {score:.3f}. Immediate containment required.",
            "attack_narrative": f"Node shows anomalous behavior primarily driven by {dominant} (SHAP: {shap[dominant]:.3f}). This suggests active exploitation or compromise in progress.",
            "mitre_tactics": ["Initial Access (TA0001)", "Execution (TA0002)", "Privilege Escalation (TA0004)", "Defense Evasion (TA0005)"],
            "remediation_steps": [
                "🚨 ISOLATE the node immediately from the network",
                "Capture forensic artifacts and memory dump",
                "Revoke all credentials and sessions",
                "Initiate incident response protocol",
                "Scan for lateral movement indicators"
            ],
            "estimated_blast_radius": f"High - potentially affects {int(score * 15)} related resources"
        }
    elif severity == "High":
        output = {
            "summary": f"⚠️ HIGH: Node {node_id} ({node_type}) shows significant threat indicators (score: {score:.3f}). Investigate within 1 hour.",
            "attack_narrative": f"Elevated risk detected with {dominant} as primary driver (SHAP: {shap[dominant]:.3f}). Patterns consistent with { 'lateral movement' if dominant == 'phi_log' else 'vulnerability exploitation' if dominant == 'phi_cve' else 'identity compromise'}.",
            "mitre_tactics": ["Lateral Movement (TA0008)", "Collection (TA0009)", "Command and Control (TA0011)"],
            "remediation_steps": [
                "Increase monitoring and logging verbosity",
                "Review recent network connections and authentication logs",
                "Check for unauthorized data access or exfiltration",
                "Verify access controls and privileges",
                "Prepare containment if escalation occurs"
            ],
            "estimated_blast_radius": f"Medium - affects approximately {int(score * 10)} neighboring resources"
        }
    elif severity == "Medium":
        output = {
            "summary": f"📊 MEDIUM: Node {node_id} ({node_type}) has elevated risk (score: {score:.3f}). Monitor closely.",
            "attack_narrative": f"Moderate anomalies detected in {dominant} (SHAP: {shap[dominant]:.3f}). Further investigation recommended to rule out early-stage compromise.",
            "mitre_tactics": ["Discovery (TA0007)", "Credential Access (TA0006)"],
            "remediation_steps": [
                "Review logs for this node over past 48 hours",
                "Check for unusual access patterns",
                "Verify user behavior analytics",
                "Document findings for future reference"
            ],
            "estimated_blast_radius": f"Limited to immediate resource group"
        }
    else:
        output = {
            "summary": f"ℹ️ LOW: Node {node_id} ({node_type}) has minimal threat indicators (score: {score:.3f}). Routine monitoring sufficient.",
            "attack_narrative": "No significant anomalies detected in the node's behavioral patterns.",
            "mitre_tactics": [],
            "remediation_steps": [
                "Continue routine monitoring",
                "No immediate action required",
                "Include in regular security reviews"
            ],
            "estimated_blast_radius": "None"
        }
    
    training_pairs.append({
        "instruction": system_prompt,
        "input": json.dumps(input_data, indent=2),
        "output": json.dumps(output, indent=2)
    })

# Save as JSONL
output_path = Path("D:\DATA PIPELINE\pipeline api\stage9_10\stage10_training_data.jsonl")
with open(output_path, "w") as f:
    for pair in training_pairs:
        f.write(json.dumps(pair) + "\n")

print(f"✅ Generated {len(training_pairs)} training pairs")
print(f"   Saved to: {output_path}")

# Show severity distribution
severity_counts = {}
for pair in training_pairs:
    inp = json.loads(pair["input"])
    sev = inp["severity"]
    severity_counts[sev] = severity_counts.get(sev, 0) + 1

print(f"\n Severity distribution:")
for sev, count in severity_counts.items():
    print(f"   {sev}: {count}")