import os, sys, torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 1 - Log Normalisation")
validate = get_api_key_validator("1")
_model   = None


def get_model():
    global _model
    if _model is not None:
        return _model

    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel

    MODEL_REPO = "Final-year-grp24/native-log-translator"
    BASE_MODEL = "google-t5/t5-base"

    print(f"Loading {MODEL_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
    base      = T5ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,   # CPU — float32 only
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, MODEL_REPO)
    model.eval()

    _model = {"tokenizer": tokenizer, "model": model}
    print("✓ Model ready")
    return _model


def classify_log(log_text: str) -> dict:
    """Use the T5 model to classify event_type, provider, risk_level."""
    try:
        m       = get_model()
        inputs  = m["tokenizer"](
            log_text, return_tensors="pt",
            max_length=128, truncation=True
        )
        with torch.no_grad():
            out = m["model"].generate(
                **inputs,
                max_new_tokens=64,
                num_beams=5,
                early_stopping=True,
            )
        decoded = m["tokenizer"].decode(out[0], skip_special_tokens=True)
        # Parse output — format is "event_type / provider / risk_level"
        parts = [p.strip() for p in decoded.split("/")]
        return {
            "event_type": parts[0] if len(parts) > 0 else "unknown",
            "provider":   parts[1] if len(parts) > 1 else "unknown",
            "risk_level": parts[2] if len(parts) > 2 else "unknown",
            "raw_output": decoded,
        }
    except Exception as e:
        return {
            "event_type": "unknown",
            "provider":   "unknown",
            "risk_level": "unknown",
            "raw_output": str(e),
        }


def extract_fields(provider: str, log: dict) -> dict:
    """Rule-based field extraction — fast and reliable for structured logs."""
    if provider == "AWS":
        return {
            "entity_id":     (log.get("userIdentity") or {}).get("userName", ""),
            "action":        log.get("eventName", ""),
            "source_ip":     log.get("sourceIPAddress", ""),
            "region":        log.get("awsRegion", ""),
            "cloud_account": log.get("recipientAccountId", ""),
            "status":        "Failed" if log.get("errorCode") else "Success",
            "target_id":     (log.get("requestParameters") or {}).get("roleName", ""),
        }
    elif provider == "Azure":
        return {
            "entity_id":     (log.get("identity") or {}).get("claims", {}).get("upn", ""),
            "action":        log.get("operationName", ""),
            "source_ip":     (log.get("properties") or {}).get("ipAddress", ""),
            "region":        log.get("location", ""),
            "cloud_account": log.get("subscriptionId", ""),
            "status":        "Success" if str(log.get("resultType", "")).lower() == "success" else "Failed",
            "target_id":     (log.get("properties") or {}).get("targetResources", [{}])[0].get("id", "") if log.get("properties", {}).get("targetResources") else "",
        }
    elif provider == "GCP":
        proto = log.get("protoPayload") or {}
        return {
            "entity_id":     proto.get("authenticationInfo", {}).get("principalEmail", ""),
            "action":        proto.get("methodName", ""),
            "source_ip":     proto.get("requestMetadata", {}).get("callerIp", ""),
            "region":        (log.get("resource") or {}).get("labels", {}).get("location", ""),
            "cloud_account": (log.get("resource") or {}).get("labels", {}).get("project_id", ""),
            "status":        "Failed" if proto.get("status", {}).get("code", 0) != 0 else "Success",
            "target_id":     proto.get("resourceName", ""),
        }
    return {
        "entity_id": "", "action": "", "source_ip": "",
        "region": "", "cloud_account": "", "status": "Unknown", "target_id": "",
    }


def log_to_text(provider: str, log: dict) -> str:
    """Convert a raw log dict to a flat text string for the T5 model."""
    if provider == "AWS":
        return (
            f"CloudTrail | eventName={log.get('eventName','')} "
            f"| userIdentity={log.get('userIdentity',{}).get('type','')} "
            f"| sourceIPAddress={log.get('sourceIPAddress','')} "
            f"| errorCode={log.get('errorCode','none')}"
        )
    elif provider == "Azure":
        return (
            f"AzureActivityLog | operationName={log.get('operationName','')} "
            f"| resultType={log.get('resultType','')} "
            f"| location={log.get('location','')}"
        )
    elif provider == "GCP":
        proto = log.get("protoPayload") or {}
        return (
            f"GCPAuditLog | methodName={proto.get('methodName','')} "
            f"| principalEmail={proto.get('authenticationInfo',{}).get('principalEmail','')} "
            f"| serviceName={proto.get('serviceName','')}"
        )
    return str(log)[:128]


class NormaliseRequest(BaseModel):
    provider:   str
    raw_log:    dict
    use_model:  bool = True   # set False to skip model and use rule-based only


class NormaliseResponse(BaseModel):
    provider:      str
    entity_type:   str
    entity_id:     str
    action:        str
    target_type:   str
    target_id:     str
    source_ip:     str
    region:        str
    cloud_account: str
    status:        str
    event_type:    str
    risk_level:    str
    model_output:  Optional[str] = None


@app.get("/health")
async def health():
    return {
        "stage":  "1",
        "status": "ok",
        "model":  "Final-year-grp24/native-log-translator",
        "base":   "google-t5/t5-base + LoRA",
        "note":   "model loads on first request (~30s cold start)",
    }


@app.post("/normalise", response_model=NormaliseResponse)
async def normalise(req: NormaliseRequest, _=Depends(validate)):
    # Step 1 — rule-based field extraction (fast, reliable)
    fields = extract_fields(req.provider, req.raw_log)

    # Step 2 — model-based classification (event_type + risk_level)
    if req.use_model:
        log_text   = log_to_text(req.provider, req.raw_log)
        classified = classify_log(log_text)
    else:
        classified = {"event_type": "unknown", "risk_level": "unknown", "raw_output": None}

    return NormaliseResponse(
        provider      = req.provider,
        entity_type   = "User",
        entity_id     = fields["entity_id"],
        action        = fields["action"],
        target_type   = "VM",
        target_id     = fields["target_id"],
        source_ip     = fields["source_ip"],
        region        = fields["region"],
        cloud_account = fields["cloud_account"],
        status        = fields["status"],
        event_type    = classified["event_type"],
        risk_level    = classified["risk_level"],
        model_output  = classified.get("raw_output"),
    )