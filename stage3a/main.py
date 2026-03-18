import os, sys, torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Optional
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 3a - Vulnerability Entity Extraction")
validate = get_api_key_validator("3A")
_model   = None


def get_model():
    global _model
    if _model is not None:
        return _model

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    MODEL_REPO = "Swapnanil09/vulnerability-extractor"
    print(f"Loading {MODEL_REPO}...")

    hf_token  = os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO, token=hf_token if hf_token else None
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float32,
        token=hf_token if hf_token else None,
    )
    model.eval()
    _model = {"tokenizer": tokenizer, "model": model}
    print("✓ Model ready")
    return _model


def extract_entities(log_text: str) -> List[dict]:
    """Run NER on log text and return extracted entities."""
    m         = get_model()
    tokenizer = m["tokenizer"]
    model     = m["model"]

    inputs = tokenizer(
        log_text, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    )

    with torch.no_grad():
        outputs     = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    entities        = []
    current_entity  = None

    for token, label in zip(tokens, labels):
        if token in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_token = token.replace("Ġ", " ").replace("▁", " ").strip()

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": clean_token, "type": label[2:]}

        elif label.startswith("I-") and current_entity:
            current_entity["text"] += clean_token

        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Clean up entity text
    for e in entities:
        e["text"] = e["text"].strip()

    return [e for e in entities if e["text"]]


def extract_cve_ids(entities: List[dict]) -> List[str]:
    """Pull CVE IDs from ERROR-type entities."""
    import re
    cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    cves = []
    for e in entities:
        if e["type"] in ("ERROR", "EXPLOIT", "SOFTWARE"):
            matches = cve_pattern.findall(e["text"])
            cves.extend(m.upper() for m in matches)
    return list(set(cves))


class ExtractRequest(BaseModel):
    log_text:    str
    edge_id:     Optional[str] = ""
    scenario_id: Optional[str] = ""
    t:           Optional[int] = 0


class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]


@app.get("/health")
async def health():
    return {
        "stage":  "3a",
        "status": "ok",
        "model":  "Swapnanil09/vulnerability-extractor",
        "base":   "microsoft/codebert-base + LoRA",
        "note":   "model loads on first request (~20s cold start)",
    }


@app.post("/extract")
async def extract(req: ExtractBatchRequest, _=Depends(validate)):
    results = []

    for log in req.logs:
        try:
            entities = extract_entities(log.log_text)
            cve_ids  = extract_cve_ids(entities)

            # Group entities by type
            grouped = {}
            for e in entities:
                grouped.setdefault(e["type"], []).append(e["text"])

            results.append({
                "edge_id":     log.edge_id,
                "scenario_id": log.scenario_id,
                "t":           log.t,
                "log_text":    log.log_text[:200],
                "entities":    entities,
                "cve_ids":     cve_ids,
                "software":    grouped.get("SOFTWARE", []),
                "versions":    grouped.get("VERSION",  []),
                "exploits":    grouped.get("EXPLOIT",  []),
                "ips":         grouped.get("IP",       []),
                "users":       grouped.get("USER",     []),
                "ports":       grouped.get("PORT",     []),
                "paths":       grouped.get("PATH",     []),
                "errors":      grouped.get("ERROR",    []),
                "n_entities":  len(entities),
            })

        except Exception as e:
            results.append({
                "edge_id":    log.edge_id,
                "log_text":   log.log_text[:200],
                "error":      str(e),
                "entities":   [],
                "cve_ids":    [],
                "n_entities": 0,
            })

    return {"results": results, "total": len(results)}