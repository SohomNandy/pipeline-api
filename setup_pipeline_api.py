"""
Run this from inside D:\DATA PIPELINE\pipeline api with venv active:
    python setup_pipeline_api.py
It writes every file in the correct folder with the correct content.
"""
from pathlib import Path

ROOT = Path(__file__).parent
files = {}

# ── .gitignore ────────────────────────────────────────────────────────────────
files[".gitignore"] = """\
venv/
__pycache__/
*.pyc
.env
*.env
.DS_Store
*.log
modal_secrets/
"""

# ── shared/__init__.py ────────────────────────────────────────────────────────
files["shared/__init__.py"] = ""

# ── shared/auth.py ────────────────────────────────────────────────────────────
files["shared/auth.py"] = """\
import os, secrets, hashlib
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_validator(stage: str):
    \"\"\"Returns a FastAPI dependency that validates X-API-Key for the given stage.\"\"\"
    env_var = f"STAGE_{stage}_API_KEY"

    async def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get(env_var, "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail=f"API key not configured for stage {stage}"
            )
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    return validate
"""

# ── gateway/requirements.txt ──────────────────────────────────────────────────
files["gateway/requirements.txt"] = """\
fastapi==0.111.0
uvicorn==0.30.1
httpx==0.27.0
python-multipart==0.0.9
pydantic==2.7.0
"""

# ── gateway/render.yaml ───────────────────────────────────────────────────────
files["gateway/render.yaml"] = """\
services:
  - type: web
    name: pipeline-gateway
    env: python
    rootDir: gateway
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: STAGE_GATEWAY_API_KEY
        generateValue: true
      - key: STAGE_0B_URL
        sync: false
      - key: STAGE_0B_API_KEY
        sync: false
      - key: STAGE_1_URL
        sync: false
      - key: STAGE_1_API_KEY
        sync: false
      - key: STAGE_2_URL
        sync: false
      - key: STAGE_2_API_KEY
        sync: false
      - key: STAGE_3B_URL
        sync: false
      - key: STAGE_3B_API_KEY
        sync: false
      - key: STAGE_4_URL
        sync: false
      - key: STAGE_4_API_KEY
        sync: false
"""

# ── gateway/main.py ───────────────────────────────────────────────────────────
files["gateway/main.py"] = """\
import os, sys
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
sys.path.append("..")
from shared.auth import get_api_key_validator

app = FastAPI(title="Pipeline API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STAGE_URLS = {
    "0B": os.environ.get("STAGE_0B_URL", ""),
    "1":  os.environ.get("STAGE_1_URL",  ""),
    "2":  os.environ.get("STAGE_2_URL",  ""),
    "3B": os.environ.get("STAGE_3B_URL", ""),
    "4":  os.environ.get("STAGE_4_URL",  ""),
}

validate_master = get_api_key_validator("GATEWAY")


async def proxy(stage: str, path: str, payload: dict) -> dict:
    url = STAGE_URLS.get(stage, "")
    if not url:
        raise HTTPException(status_code=503, detail=f"Stage {stage} URL not configured")
    stage_key = os.environ.get(f"STAGE_{stage}_API_KEY", "")
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{url.rstrip('/')}/{path}",
            json=payload,
            headers={"X-API-Key": stage_key},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stages": {k: "configured" if v else "not configured" for k, v in STAGE_URLS.items()},
    }


@app.post("/stage0b/generate")
async def stage0b_generate(request: Request, _=Depends(validate_master)):
    return await proxy("0B", "generate", await request.json())


@app.post("/stage1/normalise")
async def stage1_normalise(request: Request, _=Depends(validate_master)):
    return await proxy("1", "normalise", await request.json())


@app.post("/stage2/embed")
async def stage2_embed(request: Request, _=Depends(validate_master)):
    return await proxy("2", "embed", await request.json())


@app.post("/stage3b/score")
async def stage3b_score(request: Request, _=Depends(validate_master)):
    return await proxy("3B", "score", await request.json())


@app.post("/stage4/embed")
async def stage4_embed(request: Request, _=Depends(validate_master)):
    return await proxy("4", "embed", await request.json())
"""

# ── stage0b/requirements.txt ──────────────────────────────────────────────────
files["stage0b/requirements.txt"] = """\
modal>=0.64.0
fastapi==0.111.0
pydantic==2.7.0
"""

# ── stage0b/app.py ────────────────────────────────────────────────────────────
files["stage0b/app.py"] = '''\
import modal, os, json, torch
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.43.0",
        "peft>=0.11.1",
        "bitsandbytes>=0.43.1",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece",
        "fastapi",
        "uvicorn",
        "pydantic==2.7.0",
    )
)

app = modal.App("stage0b-siem-generator", image=image)

SYSTEM_PROMPT = (
    "You are a cloud security log renderer for a research pipeline. "
    "Given a structured security event, generate ONLY the corresponding "
    "cloud provider log as a valid JSON object. "
    "Output nothing except the JSON. No explanation. No markdown fences. "
    'The JSON must include a "_pipeline_meta" field.'
)


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class SIEMGenerator:
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login as hf_login

        hf_login(token=os.environ["HF_TOKEN"])

        BASE_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ADAPTER_ID = "sohomn/siem-log-generator-llama31-8b"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_ID,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_ID, is_trainable=False)
        self.model.eval()

    @modal.method()
    def generate(self, event: dict) -> dict:
        import warnings
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(event)},
        ]
        text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        j0, j1 = response.find("{"), response.rfind("}") + 1
        if j0 < 0:
            return {"error": "no JSON in output", "raw": response[:200]}
        try:
            return {"log": json.loads(response[j0:j1])}
        except Exception as e:
            return {"error": str(e), "raw": response[j0:j1]}


@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import os, secrets, hashlib
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader

    web            = FastAPI(title="Stage 0b - SIEM Log Generator")
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_0B_API_KEY", "")
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    generator = SIEMGenerator()

    class GenerateRequest(BaseModel):
        provider:      str
        action:        str
        entity_id:     str
        target_id:     str
        source_ip:     str = ""
        region:        str = "us-east-1"
        cloud_account: str = ""
        status:        str = "Success"
        malicious:     int = 0
        attack_phase:  str = "benign"
        edge_id:       str = ""
        scenario_id:   str = ""
        t:             int = 0

    @web.get("/health")
    async def health():
        return {"stage": "0b", "status": "ok", "model": "sohomn/siem-log-generator-llama31-8b"}

    @web.post("/generate")
    async def generate(req: GenerateRequest, _=Depends(validate)):
        result = generator.generate.remote(req.dict())
        if "error" in result:
            raise HTTPException(status_code=500, detail=result)
        return result

    return web
'''

# ── stage1/requirements.txt ───────────────────────────────────────────────────
files["stage1/requirements.txt"] = """\
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.0
"""

# ── stage1/main.py ────────────────────────────────────────────────────────────
files["stage1/main.py"] = """\
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
"""

# ── stage2/requirements.txt ───────────────────────────────────────────────────
files["stage2/requirements.txt"] = """\
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.0
transformers>=4.43.0
torch==2.3.0
"""

# ── stage2/main.py ────────────────────────────────────────────────────────────
files["stage2/main.py"] = """\
import os, sys, torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 2 - Log Embeddings")
validate = get_api_key_validator("2")
_model   = None


def get_model():
    global _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        tok  = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        mdl  = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        mdl.eval()
        proj = torch.nn.Linear(1024, 256, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        _model = {"tok": tok, "mdl": mdl, "proj": proj}
    return _model


@app.on_event("startup")
async def startup():
    get_model()


class EmbedRequest(BaseModel):
    entity_id: str
    log_texts: List[str]


@app.get("/health")
async def health():
    return {
        "stage": "2",
        "status": "stub",
        "note": "LoRA adapters not yet trained - using base BGE-Large",
    }


@app.post("/embed")
async def embed(req: EmbedRequest, _=Depends(validate)):
    m      = get_model()
    text   = "query: " + " [SEP] ".join(req.log_texts[:20])
    inputs = m["tok"](text, return_tensors="pt", truncation=True,
                      max_length=512, padding=True)
    with torch.no_grad():
        out  = m["mdl"](**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb  = torch.nn.functional.normalize(emb, dim=-1)
        z    = m["proj"](emb)[0]
    return {
        "entity_id": req.entity_id,
        "z_log":     z.tolist(),
        "dim":       256,
        "_stub":     True,
    }
"""

# ── stage3b/requirements.txt ──────────────────────────────────────────────────
files["stage3b/requirements.txt"] = """\
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.0
transformers>=4.43.0
torch==2.3.0
huggingface_hub>=0.23.0
sentencepiece
"""

# ── stage3b/main.py ───────────────────────────────────────────────────────────
files["stage3b/main.py"] = """\
import os, sys, torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 3b - CVE Risk Scoring")
validate = get_api_key_validator("3B")
_model   = None


def get_model():
    global _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        mdl = AutoModel.from_pretrained("intfloat/e5-large-v2")
        mdl.eval()
        proj = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 128),
        )
        risk_head = torch.nn.Linear(128, 2)
        _model = {"tok": tok, "mdl": mdl, "proj": proj, "risk": risk_head}
    return _model


@app.on_event("startup")
async def startup():
    get_model()


class CVERequest(BaseModel):
    cve_id:      str
    description: str


class CVEBatchRequest(BaseModel):
    cves: List[CVERequest]


@app.get("/health")
async def health():
    return {
        "stage": "3b",
        "status": "ok",
        "model": "intfloat/e5-large-v2 + projection head",
    }


@app.post("/score")
async def score(req: CVEBatchRequest, _=Depends(validate)):
    m       = get_model()
    results = []

    for cve in req.cves:
        text   = f"query: {cve.description}"
        inputs = m["tok"](text, return_tensors="pt", truncation=True,
                          max_length=512, padding=True)
        with torch.no_grad():
            out  = m["mdl"](**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = torch.nn.functional.normalize(emb, dim=-1)
            z    = m["proj"](emb)
            raw  = m["risk"](z)
            risk = torch.sigmoid(raw[0][0]).item() * 10
            expl = torch.sigmoid(raw[0][1]).item()

        results.append({
            "cve_id":       cve.cve_id,
            "z_cve":        z[0].tolist(),
            "risk_score":   round(risk, 4),
            "exploit_prob": round(expl, 4),
            "dim":          128,
        })

    return {"results": results}
"""

# ── stage4/requirements.txt ───────────────────────────────────────────────────
files["stage4/requirements.txt"] = """\
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.0
"""

# ── stage4/main.py ────────────────────────────────────────────────────────────
files["stage4/main.py"] = """\
import os, sys, hashlib
from fastapi import FastAPI, Depends
from pydantic import BaseModel
sys.path.append("..")
from shared.auth import get_api_key_validator

app      = FastAPI(title="Stage 4 - Cross-Cloud Identity Embeddings")
validate = get_api_key_validator("4")


class IdentityRequest(BaseModel):
    identity: str
    provider: str


@app.get("/health")
async def health():
    return {"stage": "4", "status": "stub", "note": "Contrastive model not yet trained"}


@app.post("/embed")
async def embed(req: IdentityRequest, _=Depends(validate)):
    key  = f"{req.identity}@{req.provider}"
    h    = hashlib.sha256(key.encode()).digest()
    vec  = [(b / 127.5) - 1.0 for b in h]
    z_id = (vec * 4)[:128]
    norm = sum(x ** 2 for x in z_id) ** 0.5
    z_id = [x / norm for x in z_id]
    return {
        "identity":   req.identity,
        "provider":   req.provider,
        "z_identity": z_id,
        "dim":        128,
        "_stub":      True,
    }
"""

# ── README.md ─────────────────────────────────────────────────────────────────
files["README.md"] = """\
# Pipeline API

Multi-cloud threat detection pipeline REST API.

## Stages

| Stage | Route | Platform | Status |
|-------|-------|----------|--------|
| Gateway | `pipeline-gateway.onrender.com` | Render | Live |
| 0b | `/stage0b/generate` | Modal GPU T4 | Live |
| 1 | `/stage1/normalise` | Render CPU | Stub |
| 2 | `/stage2/embed` | Render CPU | Stub |
| 3b | `/stage3b/score` | Render CPU | Live |
| 4 | `/stage4/embed` | Render CPU | Stub |

## Auth

All requests require master API key in header:

```
X-API-Key: <STAGE_GATEWAY_API_KEY>
```

## Health check

```bash
curl https://pipeline-gateway.onrender.com/health
```

## Example — Stage 3b

```bash
curl -X POST https://pipeline-gateway.onrender.com/stage3b/score \\
  -H "X-API-Key: YOUR_MASTER_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"cves": [{"cve_id": "CVE-2024-9999", "description": "RCE in cloud VM agent via crafted packet"}]}'
```
"""

# ── Write all files ───────────────────────────────────────────────────────────
created = []
for rel_path, content in files.items():
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    created.append(str(path))

print(f"Created {len(created)} files:")
for p in created:
    print(f"  {p}")
print("\nDone. Next step: pip install -r gateway/requirements.txt")
