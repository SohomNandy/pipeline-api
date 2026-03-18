import os, sys
from fastapi import FastAPI, Depends
from pydantic import BaseModel
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 1 - Log Normalisation")
validate = get_api_key_validator("1")


class NormaliseRequest(BaseModel):
    provider: str
    raw_log:  dict


@app.get("/health")
async def health():
    return {"stage": "1", "status": "stub", "note": "CodeBERT model not yet trained"}


@app.post("/normalise")
async def normalise(req: NormaliseRequest, _=Depends(validate)):
    log = req.raw_log

    if req.provider == "AWS":
        entity_id = (log.get("userIdentity") or {}).get("userName", "")
        action    = log.get("eventName", "")
        source_ip = log.get("sourceIPAddress", "")
        region    = log.get("awsRegion", "")
        account   = log.get("recipientAccountId", "")
        status    = "Failed" if log.get("errorCode") else "Success"
    elif req.provider == "Azure":
        entity_id = (log.get("identity") or {}).get("claims", {}).get("upn", "")
        action    = log.get("operationName", "")
        source_ip = (log.get("properties") or {}).get("ipAddress", "")
        region    = log.get("location", "")
        account   = log.get("subscriptionId", "")
        status    = "Success" if str(log.get("resultType", "")).lower() == "success" else "Failed"
    elif req.provider == "GCP":
        proto     = log.get("protoPayload") or {}
        entity_id = proto.get("authenticationInfo", {}).get("principalEmail", "")
        action    = proto.get("methodName", "")
        source_ip = proto.get("requestMetadata", {}).get("callerIp", "")
        region    = (log.get("resource") or {}).get("labels", {}).get("location", "")
        account   = (log.get("resource") or {}).get("labels", {}).get("project_id", "")
        status    = "Failed" if proto.get("status", {}).get("code", 0) != 0 else "Success"
    else:
        entity_id = action = source_ip = region = account = ""
        status    = "Unknown"

    return {
        "provider":      req.provider,
        "entity_type":   "User",
        "entity_id":     entity_id,
        "action":        action,
        "target_type":   "VM",
        "target_id":     "",
        "source_ip":     source_ip,
        "region":        region,
        "cloud_account": account,
        "status":        status,
        "_stub":         True,
    }
