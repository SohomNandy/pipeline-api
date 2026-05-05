import requests
import json
import torch
import base64
import io
from torch_geometric.data import HeteroData

url = "https://sohomnandy--stage6-rgcn-detector-fastapi-app.modal.run/predict"
api_key = "0eee318cb3cdd21ff60831f38813bc1c8707a458e360ea46cd5c2e547c5a059b"  # Replace with your actual API key

print("=" * 60)
print("TESTING STAGE 6 - RGCN DETECTOR")
print("=" * 60)

# Create a minimal HeteroData graph
print("\n📊 Creating minimal test graph...")

graph = HeteroData()

# Add a User node with 514-dim features
graph['User'].x = torch.tensor([[0.1] * 514], dtype=torch.float32)
graph['User'].node_ids = ['test_user']

# Add a VM node
graph['VM'].x = torch.tensor([[0.2] * 514], dtype=torch.float32)
graph['VM'].node_ids = ['test_vm']

# Add an edge between them
graph['User', 'ACCESS', 'VM'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

print(f"  ✓ Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Serialize to base64
buf = io.BytesIO()
torch.save(graph, buf)
graph_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
print(f"  ✓ graph_b64 length: {len(graph_b64)} chars")

# Prepare payload
payload = {
    "graph_b64": graph_b64,
    "scenario_id": "test_scenario_001"
}

print("\n📤 Sending request to Stage 6...")
print(f"   URL: {url}")
print(f"   Payload size: {len(json.dumps(payload))} bytes")

response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key
    },
    json=payload
)

print(f"\n📊 Response Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    
    print("\n✅ SUCCESS! Stage 6 Response:")
    print(f"  Scenario ID: {result.get('scenario_id')}")
    print(f"  Nodes: {result.get('n_nodes')}")
    
    h_v = result.get('h_v', [])
    if h_v:
        print(f"  h_v shape: {len(h_v)} nodes × {len(h_v[0])} dims")
        print(f"  First node embedding (first 5 values): {h_v[0][:5]}")
    
    threat_scores = result.get('threat_scores', [])
    if threat_scores:
        print(f"  Threat scores: {threat_scores}")
        print(f"  Threat score range: {min(threat_scores):.4f} - {max(threat_scores):.4f}")
    
    node_offsets = result.get('node_offsets', {})
    if node_offsets:
        print(f"  Node offsets: {node_offsets}")
    
    edge_anomaly = result.get('edge_anomaly', {})
    print(f"  Edge anomaly entries: {len(edge_anomaly)}")
    
    print("\n✅ Stage 6 test PASSED!")
    
else:
    print(f"\n❌ FAILED: {response.text}")
    
    # Check Modal logs for more details
    print("\n🔍 Check Modal logs with:")
    print("   modal app logs stage6-rgcn-detector --tail 50")