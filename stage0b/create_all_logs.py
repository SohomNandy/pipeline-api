# create_all_logs.py
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# ============================================================
# Create minimal but realistic test logs for all 3 providers
# ============================================================

def create_minimal_logs(output_path: str = "D:/DATA PIPELINE/pipeline api/stage0b/logs/all_logs.parquet"):
    """Creates a valid all_logs.parquet file for testing Stages 1-10"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample events covering different attack phases
    logs = []
    
    # 1. AWS - Privilege Escalation (Malicious)
    logs.append({
        "eventVersion": "1.08",
        "eventName": "AssumeRole",
        "eventSource": "iam.amazonaws.com",
        "eventTime": "2025-01-15T10:30:00Z",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "compromised_user",
            "accountId": "123456789012"
        },
        "requestParameters": {
            "roleArn": "arn:aws:iam::123456789012:role/admin_role",
            "roleSessionName": "compromised_session"
        },
        "sourceIPAddress": "203.0.113.45",
        "responseElements": {
            "credentials": {"accessKeyId": "ASIAEXAMPLE"}
        },
        "_pipeline_meta": {
            "edge_id": "edge_001",
            "scenario_id": "scenario_001",
            "t": 5,
            "malicious": 1,
            "attack_phase": "privilege_escalation",
            "provider": "AWS",
            "is_cross_cloud": False
        }
    })
    
    # 2. AWS - CVE Exploitation (Malicious)
    logs.append({
        "eventVersion": "1.08",
        "eventName": "RunInstances",
        "eventSource": "ec2.amazonaws.com",
        "eventTime": "2025-01-15T10:35:00Z",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "attacker",
            "accountId": "123456789012"
        },
        "requestParameters": {
            "instanceType": "c4.large",
            "imageId": "ami-12345",
            "subnetId": "subnet-abc123"
        },
        "sourceIPAddress": "203.0.113.45",
        "_pipeline_meta": {
            "edge_id": "edge_002",
            "scenario_id": "scenario_001",
            "t": 8,
            "malicious": 1,
            "attack_phase": "cve_exploitation",
            "provider": "AWS",
            "cve_id": "CVE-2024-12345",
            "is_cross_cloud": False
        }
    })
    
    # 3. AWS - Benign Admin Access
    logs.append({
        "eventVersion": "1.08",
        "eventName": "ListBuckets",
        "eventSource": "s3.amazonaws.com",
        "eventTime": "2025-01-15T09:00:00Z",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "admin_user",
            "accountId": "123456789012"
        },
        "sourceIPAddress": "192.168.1.100",
        "requestParameters": {"bucketName": "finance-data"},
        "_pipeline_meta": {
            "edge_id": "edge_003",
            "scenario_id": "scenario_001",
            "t": 2,
            "malicious": 0,
            "attack_phase": "benign",
            "provider": "AWS",
            "is_cross_cloud": False
        }
    })
    
    # 4. Azure - Lateral Movement (Malicious)
    logs.append({
        "operationName": "MICROSOFT.COMPUTE/VIRTUALMACHINES/START/ACTION",
        "caller": "attacker@malicious.com",
        "callerIpAddress": "203.0.113.45",
        "correlationId": "abc-123-def",
        "eventTimestamp": "2025-01-15T10:40:00Z",
        "subscriptionId": "sub-azure-123",
        "properties": {
            "vmName": "victim-vm-01",
            "resourceGroup": "prod-rg"
        },
        "_pipeline_meta": {
            "edge_id": "edge_004",
            "scenario_id": "scenario_001",
            "t": 10,
            "malicious": 1,
            "attack_phase": "lateral_movement",
            "provider": "Azure",
            "is_cross_cloud": False
        }
    })
    
    # 5. Azure - Benign VM Start
    logs.append({
        "operationName": "MICROSOFT.COMPUTE/VIRTUALMACHINES/START/ACTION",
        "caller": "admin@corp.com",
        "callerIpAddress": "10.0.0.5",
        "correlationId": "def-456-ghi",
        "eventTimestamp": "2025-01-15T08:00:00Z",
        "subscriptionId": "sub-azure-123",
        "properties": {
            "vmName": "web-server-01",
            "resourceGroup": "web-rg"
        },
        "_pipeline_meta": {
            "edge_id": "edge_005",
            "scenario_id": "scenario_001",
            "t": 1,
            "malicious": 0,
            "attack_phase": "benign",
            "provider": "Azure",
            "is_cross_cloud": False
        }
    })
    
    # 6. GCP - Cross-Cloud Pivot (Malicious)
    logs.append({
        "protoPayload": {
            "methodName": "compute.instances.start",
            "authenticationInfo": {
                "principalEmail": "attacker@example.com"
            },
            "authorizationInfo": [{"granted": True}],
            "request": {"name": "victim-instance"},
            "resourceName": "projects/test-project/zones/us-central1-a/instances/victim-instance"
        },
        "timestamp": "2025-01-15T10:45:00Z",
        "logName": "projects/test-project/logs/cloudaudit.googleapis.com%2Factivity",
        "_pipeline_meta": {
            "edge_id": "edge_006",
            "scenario_id": "scenario_001",
            "t": 12,
            "malicious": 1,
            "attack_phase": "cross_cloud_pivot",
            "provider": "GCP",
            "source_provider": "AWS",
            "is_cross_cloud": True
        }
    })
    
    # 7. GCP - Benign Storage Access
    logs.append({
        "protoPayload": {
            "methodName": "storage.objects.get",
            "authenticationInfo": {
                "principalEmail": "service-account@project.iam.gserviceaccount.com"
            },
            "resourceName": "projects/_/buckets/my-bucket/objects/file.txt"
        },
        "timestamp": "2025-01-15T07:00:00Z",
        "_pipeline_meta": {
            "edge_id": "edge_007",
            "scenario_id": "scenario_001",
            "t": 0,
            "malicious": 0,
            "attack_phase": "benign",
            "provider": "GCP",
            "is_cross_cloud": False
        }
    })
    
    # Create DataFrame and save
    df = pd.DataFrame(logs)
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Created {output_path}")
    print(f"   Total logs: {len(df)}")
    print(f"   Providers: {df['_pipeline_meta'].apply(lambda x: x['provider']).value_counts().to_dict()}")
    print(f"   Attack phases: {df['_pipeline_meta'].apply(lambda x: x['attack_phase']).value_counts().to_dict()}")
    print(f"   Malicious count: {sum(1 for x in df['_pipeline_meta'] if x['malicious'] == 1)}")
    print(f"   Benign count: {sum(1 for x in df['_pipeline_meta'] if x['malicious'] == 0)}")
    
    return df

# ============================================================
# Also create individual provider files (optional)
# ============================================================
def create_provider_specific_files(base_path: str = "D:/DATA PIPELINE/pipeline api/stage0b/logs"):
    """Creates aws_logs.parquet, azure_logs.parquet, gcp_logs.parquet"""
    
    os.makedirs(base_path, exist_ok=True)
    
    # Read the main file
    df = pd.read_parquet(os.path.join(base_path, "all_logs.parquet"))
    
    # Split by provider
    aws_logs = []
    azure_logs = []
    gcp_logs = []
    
    for _, row in df.iterrows():
        meta = row['_pipeline_meta']
        provider = meta['provider']
        
        # Create a clean copy without the nested meta (or keep it)
        log_entry = row.to_dict()
        
        if provider == 'AWS':
            aws_logs.append(log_entry)
        elif provider == 'Azure':
            azure_logs.append(log_entry)
        else:
            gcp_logs.append(log_entry)
    
    # Save individual files
    if aws_logs:
        pd.DataFrame(aws_logs).to_parquet(os.path.join(base_path, "aws_logs.parquet"), index=False)
        print(f"   Saved {len(aws_logs)} AWS logs")
    if azure_logs:
        pd.DataFrame(azure_logs).to_parquet(os.path.join(base_path, "azure_logs.parquet"), index=False)
        print(f"   Saved {len(azure_logs)} Azure logs")
    if gcp_logs:
        pd.DataFrame(gcp_logs).to_parquet(os.path.join(base_path, "gcp_logs.parquet"), index=False)
        print(f"   Saved {len(gcp_logs)} GCP logs")

# ============================================================
# Run the script
# ============================================================
if __name__ == "__main__":
    # Create main file
    df = create_minimal_logs("D:/DATA PIPELINE/pipeline api/stage0b/logs/all_logs.parquet")
    
    # Create provider-specific files (optional)
    create_provider_specific_files("D:/DATA PIPELINE/pipeline api/stage0b/logs")
    
    print("\n✅ All log files created successfully!")
    print("\nFile locations:")
    print("   - D:/DATA PIPELINE/pipeline api/stage0b/logs/all_logs.parquet")
    print("   - D:/DATA PIPELINE/pipeline api/stage0b/logs/aws_logs.parquet")
    print("   - D:/DATA PIPELINE/pipeline api/stage0b/logs/azure_logs.parquet")
    print("   - D:/DATA PIPELINE/pipeline api/stage0b/logs/gcp_logs.parquet")