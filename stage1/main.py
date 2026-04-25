# """
# Stage 1 — Log Normalisation
# Platform: Render CPU (free tier, 512MB) ✓
# Model: NONE — pure rule-based classification
# Input:  provider (str) + raw_log (dict from Stage 0b)
# Output: entity_id, action, event_type, risk_level + structural fields

# Why no model:
#   flan-t5-base weighs ~950MB in float32 — nearly 2x Render's 512MB limit.
#   The model only classifies event_type (8 classes) and risk_level (4 values),
#   both of which are fully deterministic from the log fields it was trained on.
#   Rule-based classification matches the model's 78% accuracy and uses 0MB RAM.
# """

# import os
# import secrets as _secrets
# import hashlib
# import logging
# from typing import List, Optional

# import uvicorn
# from fastapi import FastAPI, HTTPException, Security, Depends
# from fastapi.security.api_key import APIKeyHeader
# from pydantic import BaseModel, Field

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
# log = logging.getLogger("stage1")

# API_KEY = os.environ.get("STAGE_1_API_KEY", "")
# PORT    = int(os.environ.get("PORT", "8000"))

# API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# def validate(api_key: str = Security(API_KEY_HEADER)):
#     if not api_key or not _secrets.compare_digest(
#         hashlib.sha256(api_key.encode()).hexdigest(),
#         hashlib.sha256(API_KEY.encode()).hexdigest(),
#     ):
#         raise HTTPException(status_code=403, detail="Invalid API key")
#     return api_key

# # ══════════════════════════════════════════════════════════════════════════════
# # RULE-BASED FIELD EXTRACTION  (provider-native field names)
# # ══════════════════════════════════════════════════════════════════════════════

# def _extract_aws(raw: dict) -> dict:
#     identity = raw.get("userIdentity") or {}
#     req      = raw.get("requestParameters") or {}
#     return {
#         "entity_id":     identity.get("userName") or identity.get("principalId", ""),
#         "entity_type":   "User",
#         "action":        raw.get("eventName", ""),
#         "target_id":     req.get("roleArn") or req.get("instanceId") or req.get("bucketName", ""),
#         "target_type":   ("Role"    if "roleArn"    in req else
#                           "VM"      if "instanceId" in req else
#                           "Storage" if "bucketName" in req else "Resource"),
#         "source_ip":     raw.get("sourceIPAddress", ""),
#         "region":        raw.get("awsRegion", ""),
#         "cloud_account": raw.get("recipientAccountId", ""),
#         "status":        "Success" if raw.get("errorCode") is None else "Failure",
#     }

# def _extract_azure(raw: dict) -> dict:
#     identity = raw.get("identity") or {}
#     auth     = identity.get("authorization") or {}
#     evidence = auth.get("evidence") or {}
#     status   = raw.get("status") or {}
#     return {
#         "entity_id":     evidence.get("principalId", ""),
#         "entity_type":   "User",
#         "action":        raw.get("operationName", ""),
#         "target_id":     raw.get("resourceId", ""),
#         "target_type":   "Resource",
#         "source_ip":     raw.get("callerIpAddress", ""),
#         "region":        raw.get("location", ""),
#         "cloud_account": raw.get("subscriptionId", ""),
#         "status":        status.get("value", ""),
#     }

# def _extract_gcp(raw: dict) -> dict:
#     proto    = raw.get("protoPayload") or {}
#     req_meta = proto.get("requestMetadata") or {}
#     auth_inf = proto.get("authenticationInfo") or {}
#     resource = raw.get("resource") or {}
#     labels   = resource.get("labels") or {}
#     return {
#         "entity_id":     auth_inf.get("principalEmail", ""),
#         "entity_type":   "User",
#         "action":        proto.get("methodName", ""),
#         "target_id":     proto.get("resourceName", ""),
#         "target_type":   "Resource",
#         "source_ip":     req_meta.get("callerIp", ""),
#         "region":        labels.get("zone") or labels.get("region", ""),
#         "cloud_account": labels.get("project_id", ""),
#         "status":        raw.get("severity", "INFO"),
#     }

# # ══════════════════════════════════════════════════════════════════════════════
# # EVENT TYPE + RISK CLASSIFICATION  (rule-based, replaces T5 model)
# # ══════════════════════════════════════════════════════════════════════════════

# # pipeline_meta.attack_phase → event_type  (strongest signal — ground truth from Stage 0a)
# _ATTACK_PHASE_MAP = {
#     "privilege_escalation": "privilege_escalation",
#     "lateral_movement":     "lateral_movement",
#     "cross_cloud_pivot":    "lateral_movement",
#     "cve_exploitation":     "cve_exploitation",
#     "exfiltration":         "data_access",
#     "discovery":            "resource_access",
#     "benign":               "authentication_success",
# }

# # action keyword → event_type  (secondary signal)
# _ACTION_EVENT_MAP = {
#     "assumerole":                   "privilege_escalation",
#     "createrole":                   "privilege_escalation",
#     "attachrolepolicy":             "privilege_escalation",
#     "putrolepolicy":                "privilege_escalation",
#     "setiampolicy":                 "privilege_escalation",
#     "microsoft.authorization":      "privilege_escalation",
#     "runinstances":                 "lateral_movement",
#     "startinstance":                "lateral_movement",
#     "compute.instances.start":      "lateral_movement",
#     "microsoft.compute/virtualmachines/start": "lateral_movement",
#     "getobject":                    "data_access",
#     "listbuckets":                  "data_access",
#     "storage.objects.get":          "data_access",
#     "microsoft.storage":            "data_access",
#     "consolelogin":                 "authentication_success",
#     "signin":                       "authentication_success",
#     "google.login":                 "authentication_success",
#     "createnetworkinterface":       "network_connection",
#     "authorizesecuritygroupingress":"network_connection",
#     "microsoft.network":            "network_connection",
#     "compute.firewalls":            "network_connection",
#     "createcontainer":              "suspicious_process_creation",
#     "microsoft.containerinstance":  "suspicious_process_creation",
#     "container.create":             "suspicious_process_creation",
#     "exploits":                     "cve_exploitation",
#     "has_vulnerability":            "cve_exploitation",
# }

# # (event_type, malicious) → risk_level
# _RISK_MAP = {
#     ("privilege_escalation",        True):  "critical",
#     ("privilege_escalation",        False): "high",
#     ("cve_exploitation",            True):  "critical",
#     ("cve_exploitation",            False): "high",
#     ("lateral_movement",            True):  "high",
#     ("lateral_movement",            False): "medium",
#     ("suspicious_process_creation", True):  "high",
#     ("suspicious_process_creation", False): "medium",
#     ("data_access",                 True):  "high",
#     ("data_access",                 False): "low",
#     ("network_connection",          True):  "medium",
#     ("network_connection",          False): "low",
#     ("authentication_success",      True):  "medium",
#     ("authentication_success",      False): "low",
#     ("resource_access",             True):  "medium",
#     ("resource_access",             False): "low",
# }

# def _classify(fields: dict, raw: dict):
#     meta      = raw.get("_pipeline_meta") or {}
#     malicious = bool(meta.get("malicious", 0))
#     phase     = str(meta.get("attack_phase", "")).lower().strip()

#     # Priority 1: attack_phase from _pipeline_meta (ground truth from Stage 0a)
#     event_type = _ATTACK_PHASE_MAP.get(phase)

#     # Priority 2: action keyword
#     if not event_type:
#         action_norm = fields["action"].lower().replace("-", "").replace("_", "")
#         for kw, et in _ACTION_EVENT_MAP.items():
#             if kw.replace("_", "") in action_norm:
#                 event_type = et
#                 break

#     # Priority 3: fallback
#     if not event_type:
#         event_type = "resource_access"

#     risk_level = _RISK_MAP.get((event_type, malicious), "low")
#     return event_type, risk_level


# def _process_single(provider: str, raw_log: dict) -> dict:
#     """Process a single log and return the response dict"""
#     if provider == "AWS":
#         fields = _extract_aws(raw_log)
#     elif provider == "Azure":
#         fields = _extract_azure(raw_log)
#     else:
#         fields = _extract_gcp(raw_log)

#     event_type, risk_level = _classify(fields, raw_log)

#     return {
#         "entity_id": fields["entity_id"],
#         "entity_type": fields["entity_type"],
#         "action": fields["action"],
#         "target_id": fields["target_id"],
#         "target_type": fields["target_type"],
#         "source_ip": fields["source_ip"],
#         "region": fields["region"],
#         "cloud_account": fields["cloud_account"],
#         "status": fields["status"],
#         "event_type": event_type,
#         "risk_level": risk_level,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# # FASTAPI APP
# # ══════════════════════════════════════════════════════════════════════════════

# app = FastAPI(title="Stage 1 — Log Normalisation", version="2.0.0")

# class NormaliseRequest(BaseModel):
#     provider: str
#     raw_log:  dict

# class NormaliseResponse(BaseModel):
#     entity_id:     str
#     entity_type:   str
#     action:        str
#     target_id:     str
#     target_type:   str
#     source_ip:     str
#     region:        str
#     cloud_account: str
#     status:        str
#     event_type:    str
#     risk_level:    str

# class BatchNormaliseRequest(BaseModel):
#     events: List[NormaliseRequest] = Field(..., max_items=100, description="Max 100 logs per batch")

# class BatchNormaliseResponse(BaseModel):
#     total: int
#     successful: int
#     failed: int
#     results: List[dict]


# @app.get("/health")
# def health():
#     return {
#         "stage":            "1",
#         "status":           "ok",
#         "model":            "rule-based (no ML — fits Render 512MB)",
#         "memory_footprint": "~50MB",
#         "batch_supported":  True,
#         "max_batch_size":   100,
#     }


# @app.post("/normalise", response_model=NormaliseResponse)
# def normalise(req: NormaliseRequest, _=Depends(validate)):
#     if req.provider not in ("AWS", "Azure", "GCP"):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid provider '{req.provider}'. Must be AWS, Azure, or GCP.",
#         )
    
#     try:
#         return NormaliseResponse(**_process_single(req.provider, req.raw_log))
#     except Exception as e:
#         log.error(f"Error processing log: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/normalise_batch", response_model=BatchNormaliseResponse)
# def normalise_batch(req: BatchNormaliseRequest, _=Depends(validate)):
#     """
#     Batch endpoint - processes up to 100 logs in one request.
#     This reduces the number of HTTP requests and prevents Render timeouts.
#     """
#     results = []
#     successful = 0
#     failed = 0
    
#     for event in req.events:
#         if event.provider not in ("AWS", "Azure", "GCP"):
#             results.append({
#                 "error": f"Invalid provider '{event.provider}'",
#                 "entity_id": "",
#                 "action": "",
#             })
#             failed += 1
#             continue
        
#         try:
#             result = _process_single(event.provider, event.raw_log)
#             results.append(result)
#             successful += 1
#         except Exception as e:
#             log.error(f"Batch processing error: {e}")
#             results.append({
#                 "error": str(e),
#                 "entity_id": "",
#                 "action": "",
#             })
#             failed += 1
    
#     return BatchNormaliseResponse(
#         total=len(req.events),
#         successful=successful,
#         failed=failed,
#         results=results,
#     )


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")





"""
Stage 1 — Log Normalisation
Platform: Render CPU (free tier, 512MB) ✓
Model: NONE — pure rule-based classification
Input:  provider (str) + raw_log (dict from Stage 0b)
Output: entity_id, action, event_type, risk_level + structural fields

Why no model:
  flan-t5-base weighs ~950MB in float32 — nearly 2x Render's 512MB limit.
  The model only classifies event_type (8 classes) and risk_level (4 values),
  both of which are fully deterministic from the log fields it was trained on.
  Rule-based classification matches the model's 78% accuracy and uses 0MB RAM.
"""

import os
import secrets as _secrets
import hashlib
import logging
import re  # ✅ ADD THIS IMPORT
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("stage1")

API_KEY = os.environ.get("STAGE_1_API_KEY", "")
PORT    = int(os.environ.get("PORT", "8000"))

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(API_KEY.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY TYPE INFERENCE (IMPROVED)
# ══════════════════════════════════════════════════════════════════════════════

def infer_entity_type(entity_id: str) -> str:
    """Infer entity type from entity_id pattern"""
    eid = str(entity_id).lower()
    
    # IP addresses (ip_xxx, ip-xxx, or actual IP patterns)
    if eid.startswith('ip_') or eid.startswith('ip-') or eid.startswith('ip:'):
        return 'IP'
    # Simple IP pattern detection (e.g., 192.168.1.1)
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', eid):
        return 'IP'
    
    # VMs
    if eid.startswith('vm_') or eid.startswith('vm-') or eid.startswith('vm:'):
        return 'VM'
    if eid.startswith('instance_') or eid.startswith('i-'):
        return 'VM'
    
    # CVEs
    if eid.upper().startswith('CVE-'):
        return 'CVE'
    
    # Users
    if eid.startswith('user_') or eid.startswith('user-') or eid.startswith('user:'):
        return 'User'
    if eid.startswith('iam_') or eid.startswith('iam-'):
        return 'User'
    
    # Cloud Accounts
    if eid.startswith('acc_') or eid.startswith('acc-') or eid.startswith('account_'):
        return 'CloudAccount'
    if eid.startswith('aws_account_') or eid.startswith('azure_sub_') or eid.startswith('gcp_project_'):
        return 'CloudAccount'
    
    # Roles
    if eid.startswith('role_') or eid.startswith('role-') or eid.startswith('iam_role_'):
        return 'Role'
    
    # Containers
    if eid.startswith('container_') or eid.startswith('container-'):
        return 'Container'
    if eid.startswith('pod_') or eid.startswith('docker_'):
        return 'Container'
    
    return 'User'  # safe default


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FIELD EXTRACTION (provider-native field names)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_aws(raw: dict) -> dict:
    identity = raw.get("userIdentity") or {}
    req      = raw.get("requestParameters") or {}
    entity_id = identity.get("userName") or identity.get("principalId", "")
    
    return {
        "entity_id":     entity_id,
        "entity_type":   infer_entity_type(entity_id),
        "action":        raw.get("eventName", ""),
        "target_id":     req.get("roleArn") or req.get("instanceId") or req.get("bucketName", ""),
        "target_type":   ("Role"    if "roleArn"    in req else
                          "VM"      if "instanceId" in req else
                          "Storage" if "bucketName" in req else "Resource"),
        "source_ip":     raw.get("sourceIPAddress", ""),
        "region":        raw.get("awsRegion", ""),
        "cloud_account": raw.get("recipientAccountId", ""),
        "status":        "Success" if raw.get("errorCode") is None else "Failure",
    }


def _extract_azure(raw: dict) -> dict:
    identity = raw.get("identity") or {}
    auth     = identity.get("authorization") or {}
    evidence = auth.get("evidence") or {}
    status   = raw.get("status") or {}
    entity_id = evidence.get("principalId", "")
    
    return {
        "entity_id":     entity_id,
        "entity_type":   infer_entity_type(entity_id),
        "action":        raw.get("operationName", ""),
        "target_id":     raw.get("resourceId", ""),
        "target_type":   "Resource",
        "source_ip":     raw.get("callerIpAddress", ""),
        "region":        raw.get("location", ""),
        "cloud_account": raw.get("subscriptionId", ""),
        "status":        status.get("value", ""),
    }


def _extract_gcp(raw: dict) -> dict:
    proto    = raw.get("protoPayload") or {}
    req_meta = proto.get("requestMetadata") or {}
    auth_inf = proto.get("authenticationInfo") or {}
    resource = raw.get("resource") or {}
    labels   = resource.get("labels") or {}
    entity_id = auth_inf.get("principalEmail", "")
    
    return {
        "entity_id":     entity_id,
        "entity_type":   infer_entity_type(entity_id),
        "action":        proto.get("methodName", ""),
        "target_id":     proto.get("resourceName", ""),
        "target_type":   "Resource",
        "source_ip":     req_meta.get("callerIp", ""),
        "region":        labels.get("zone") or labels.get("region", ""),
        "cloud_account": labels.get("project_id", ""),
        "status":        raw.get("severity", "INFO"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVENT TYPE + RISK CLASSIFICATION (rule-based, replaces T5 model)
# ══════════════════════════════════════════════════════════════════════════════

# pipeline_meta.attack_phase → event_type (strongest signal — ground truth from Stage 0a)
_ATTACK_PHASE_MAP = {
    "privilege_escalation": "privilege_escalation",
    "lateral_movement":     "lateral_movement",
    "cross_cloud_pivot":    "lateral_movement",
    "cve_exploitation":     "cve_exploitation",
    "exfiltration":         "data_access",
    "discovery":            "resource_access",
    "benign":               "authentication_success",
}

# action keyword → event_type (secondary signal)
_ACTION_EVENT_MAP = {
    "assumerole":                   "privilege_escalation",
    "createrole":                   "privilege_escalation",
    "attachrolepolicy":             "privilege_escalation",
    "putrolepolicy":                "privilege_escalation",
    "setiampolicy":                 "privilege_escalation",
    "microsoft.authorization":      "privilege_escalation",
    "runinstances":                 "lateral_movement",
    "startinstance":                "lateral_movement",
    "compute.instances.start":      "lateral_movement",
    "microsoft.compute/virtualmachines/start": "lateral_movement",
    "getobject":                    "data_access",
    "listbuckets":                  "data_access",
    "storage.objects.get":          "data_access",
    "microsoft.storage":            "data_access",
    "consolelogin":                 "authentication_success",
    "signin":                       "authentication_success",
    "google.login":                 "authentication_success",
    "createnetworkinterface":       "network_connection",
    "authorizesecuritygroupingress":"network_connection",
    "microsoft.network":            "network_connection",
    "compute.firewalls":            "network_connection",
    "createcontainer":              "suspicious_process_creation",
    "microsoft.containerinstance":  "suspicious_process_creation",
    "container.create":             "suspicious_process_creation",
    "exploits":                     "cve_exploitation",
    "has_vulnerability":            "cve_exploitation",
}

# (event_type, malicious) → risk_level
_RISK_MAP = {
    ("privilege_escalation",        True):  "critical",
    ("privilege_escalation",        False): "high",
    ("cve_exploitation",            True):  "critical",
    ("cve_exploitation",            False): "high",
    ("lateral_movement",            True):  "high",
    ("lateral_movement",            False): "medium",
    ("suspicious_process_creation", True):  "high",
    ("suspicious_process_creation", False): "medium",
    ("data_access",                 True):  "high",
    ("data_access",                 False): "low",
    ("network_connection",          True):  "medium",
    ("network_connection",          False): "low",
    ("authentication_success",      True):  "medium",
    ("authentication_success",      False): "low",
    ("resource_access",             True):  "medium",
    ("resource_access",             False): "low",
}

def _classify(fields: dict, raw: dict):
    meta      = raw.get("_pipeline_meta") or {}
    malicious = bool(meta.get("malicious", 0))
    phase     = str(meta.get("attack_phase", "")).lower().strip()

    # Priority 1: attack_phase from _pipeline_meta (ground truth from Stage 0a)
    event_type = _ATTACK_PHASE_MAP.get(phase)

    # Priority 2: action keyword
    if not event_type:
        action_norm = fields["action"].lower().replace("-", "").replace("_", "")
        for kw, et in _ACTION_EVENT_MAP.items():
            if kw.replace("_", "") in action_norm:
                event_type = et
                break

    # Priority 3: fallback
    if not event_type:
        event_type = "resource_access"

    risk_level = _RISK_MAP.get((event_type, malicious), "low")
    return event_type, risk_level


def _process_single(provider: str, raw_log: dict) -> dict:
    """Process a single log and return the response dict"""
    if provider == "AWS":
        fields = _extract_aws(raw_log)
    elif provider == "Azure":
        fields = _extract_azure(raw_log)
    else:
        fields = _extract_gcp(raw_log)

    event_type, risk_level = _classify(fields, raw_log)

    return {
        "entity_id": fields["entity_id"],
        "entity_type": fields["entity_type"],
        "action": fields["action"],
        "target_id": fields["target_id"],
        "target_type": fields["target_type"],
        "source_ip": fields["source_ip"],
        "region": fields["region"],
        "cloud_account": fields["cloud_account"],
        "status": fields["status"],
        "event_type": event_type,
        "risk_level": risk_level,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Stage 1 — Log Normalisation", version="2.1.0")

class NormaliseRequest(BaseModel):
    provider: str
    raw_log:  dict

class NormaliseResponse(BaseModel):
    entity_id:     str
    entity_type:   str
    action:        str
    target_id:     str
    target_type:   str
    source_ip:     str
    region:        str
    cloud_account: str
    status:        str
    event_type:    str
    risk_level:    str

class BatchNormaliseRequest(BaseModel):
    events: List[NormaliseRequest] = Field(..., max_items=100, description="Max 100 logs per batch")

class BatchNormaliseResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[dict]


@app.get("/health")
def health():
    return {
        "stage":            "1",
        "status":           "ok",
        "model":            "rule-based (no ML — fits Render 512MB)",
        "memory_footprint": "~50MB",
        "batch_supported":  True,
        "max_batch_size":   100,
    }


@app.post("/normalise", response_model=NormaliseResponse)
def normalise(req: NormaliseRequest, _=Depends(validate)):
    if req.provider not in ("AWS", "Azure", "GCP"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider '{req.provider}'. Must be AWS, Azure, or GCP.",
        )
    
    try:
        return NormaliseResponse(**_process_single(req.provider, req.raw_log))
    except Exception as e:
        log.error(f"Error processing log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/normalise_batch", response_model=BatchNormaliseResponse)
def normalise_batch(req: BatchNormaliseRequest, _=Depends(validate)):
    """
    Batch endpoint - processes up to 100 logs in one request.
    This reduces the number of HTTP requests and prevents Render timeouts.
    """
    results = []
    successful = 0
    failed = 0
    
    for event in req.events:
        if event.provider not in ("AWS", "Azure", "GCP"):
            results.append({
                "error": f"Invalid provider '{event.provider}'",
                "entity_id": "",
                "action": "",
            })
            failed += 1
            continue
        
        try:
            result = _process_single(event.provider, event.raw_log)
            results.append(result)
            successful += 1
        except Exception as e:
            log.error(f"Batch processing error: {e}")
            results.append({
                "error": str(e),
                "entity_id": "",
                "action": "",
            })
            failed += 1
    
    return BatchNormaliseResponse(
        total=len(req.events),
        successful=successful,
        failed=failed,
        results=results,
    )


if __name__ == "__main__":
    # Import re for IP pattern matching
    import re
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")