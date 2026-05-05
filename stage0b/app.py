# """
# Stage 0b — SIEM Log Generator
# Model: sohomn/siem-log-generator-llama31-8b (LLaMA 3.1 8B + QLoRA)
# Platform: Modal GPU T4
# Input:  structured event dict (provider, action, entity_id, ...)
# Output: provider-native JSON log with _pipeline_meta field
# """

# import modal, os, json
# from typing import List
# import hashlib
# import secrets as _secrets

# # ══════════════════════════════════════════════════════════════════════════════
# # MODAL IMAGE + APP
# # ══════════════════════════════════════════════════════════════════════════════

# image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .pip_install(
#         "transformers>=4.43.0",
#         "peft>=0.11.1",
#         "bitsandbytes>=0.43.1",
#         "accelerate>=0.30.0",
#         "huggingface_hub>=0.23.0",
#         "sentencepiece",
#         "fastapi",
#         "uvicorn",
#         "pydantic==2.7.0",
#         "tenacity>=8.2.0",
#     )
# )

# app = modal.App("stage0b-siem-generator", image=image)

# SYSTEM_PROMPT = (
#     "You are a cloud security log renderer for a research pipeline. "
#     "Given a structured security event, generate ONLY the corresponding "
#     "cloud provider log as a valid JSON object. "
#     "Output nothing except the JSON. No explanation. No markdown fences. "
#     'The JSON must include a "_pipeline_meta" field.'
# )

# # ── Provider-aware fallback templates ────────────────────────────────────────
# # These are used when LLM generation fails. Each matches the provider's native
# # log structure so downstream provider detection works correctly.

# def _fallback_log(event: dict) -> dict:
#     """
#     Build a minimal but structurally correct provider-native fallback log.
#     Each provider template uses the real field names for that cloud so that
#     field-sniffing in task_stage0b works as a secondary detection method.
#     """
#     provider = event.get('provider', 'AWS')
#     meta = {
#         'edge_id':      event.get('edge_id',      ''),
#         'scenario_id':  event.get('scenario_id',  ''),
#         't':            event.get('t',             0),
#         'malicious':    event.get('malicious',     0),
#         'attack_phase': event.get('attack_phase',  'benign'),
#         'provider':     provider,   # ← always stamped — primary detection signal
#         'fallback':     True,
#     }

#     if provider == 'AWS':
#         return {
#             'eventVersion':   '1.08',
#             'eventSource':    'iam.amazonaws.com',          # AWS field — detectable
#             'eventName':      event.get('action', 'AssumeRole'),
#             'awsRegion':      event.get('region', 'us-east-1'),
#             'sourceIPAddress':event.get('source_ip', '0.0.0.0'),
#             'userIdentity':   {'userName': event.get('entity_id', 'unknown')},
#             'requestParameters': {'roleArn': event.get('target_id', '')},
#             'responseElements':  {'assumedRoleUser': {'arn': ''}},
#             '_pipeline_meta': meta,
#         }

#     elif provider == 'Azure':
#         return {
#             'operationName':  event.get('action', 'Microsoft.Compute/virtualMachines/read'),
#             'subscriptionId': event.get('cloud_account', 'sub-00000000'),  # Azure field
#             'resourceGroup':  'rg-default',
#             'resourceId':     f"/subscriptions/{event.get('cloud_account','')}/providers/Microsoft.Compute",
#             'callerIpAddress':event.get('source_ip', '0.0.0.0'),
#             'identity':       {'authorization': {'evidence': {'principalId': event.get('entity_id','')}}},
#             'properties':     {'targetId': event.get('target_id', '')},
#             'status':         {'value': event.get('status', 'Succeeded')},
#             '_pipeline_meta': meta,
#         }

#     else:  # GCP
#         return {
#             'protoPayload': {                               # GCP field — detectable
#                 '@type':         'type.googleapis.com/google.cloud.audit.AuditLog',
#                 'serviceName':   'compute.googleapis.com',
#                 'methodName':    event.get('action', 'v1.compute.instances.get'),
#                 'authenticationInfo': {'principalEmail': event.get('entity_id', 'unknown@project.iam')},
#                 'requestMetadata':    {'callerIp': event.get('source_ip', '0.0.0.0')},
#                 'resourceName':       event.get('target_id', ''),
#             },
#             'resource': {
#                 'type':   'gce_instance',
#                 'labels': {'project_id': event.get('cloud_account', 'gcp-project')},
#             },
#             'timestamp':      '2024-01-01T00:00:00Z',
#             'severity':       'INFO',
#             '_pipeline_meta': meta,
#         }


# # ══════════════════════════════════════════════════════════════════════════════
# # SIEM GENERATOR CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# @app.cls(
#     gpu="T4",
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     container_idle_timeout=300,
#     allow_concurrent_inputs=5,
# )
# class SIEMGenerator:

#     def __enter__(self):
#         from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#         from peft import PeftModel
#         from huggingface_hub import login as hf_login
#         import torch

#         hf_token = os.environ.get("HF_TOKEN", "")
#         if hf_token:
#             hf_login(token=hf_token)
#         else:
#             print("WARNING: HF_TOKEN not set — model loading may fail for gated repos")

#         BASE_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#         ADAPTER_ID = "sohomn/siem-log-generator-llama31-8b"

#         bnb = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )

#         print(f"Loading tokenizer from {BASE_ID}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         print(f"Loading base model {BASE_ID}...")
#         base = AutoModelForCausalLM.from_pretrained(
#             BASE_ID,
#             quantization_config=bnb,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             attn_implementation="eager",
#         )

#         print(f"Loading LoRA adapter {ADAPTER_ID}...")
#         self.model = PeftModel.from_pretrained(base, ADAPTER_ID, is_trainable=False)
#         self.model.eval()

#         print("Warming up...")
#         self._generate_single({"provider": "AWS", "action": "TEST",
#                                 "entity_id": "warmup", "scenario_id": "warmup"})
#         print("Stage 0b ready ✓")

#     def _generate_single(self, event: dict, max_retries: int = 2) -> dict:
#         import torch, warnings
#         from tenacity import (retry, stop_after_attempt,
#                               wait_exponential, retry_if_exception_type)

#         @retry(
#             stop=stop_after_attempt(max_retries),
#             wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
#             retry=retry_if_exception_type(Exception),
#         )
#         def _gen():
#             messages = [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user",   "content": json.dumps(event)},
#             ]
#             text   = self.tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True)
#             inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

#             with torch.no_grad(), warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 out = self.model.generate(
#                     **inputs,
#                     max_new_tokens=512,
#                     do_sample=False,
#                     temperature=0.0,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     repetition_penalty=1.0,
#                 )

#             response = self.tokenizer.decode(
#                 out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
#             ).strip()

#             j0 = response.find("{")
#             j1 = response.rfind("}") + 1
#             if j0 < 0 or j1 <= 0:
#                 raise ValueError(f"No JSON in response: {response[:200]}")

#             json_str = (response[j0:j1]
#                         .replace("'", '"')
#                         .replace("None", "null")
#                         .replace("True", "true")
#                         .replace("False", "false"))
#             parsed = json.loads(json_str)

#             # Ensure _pipeline_meta is always present and has correct provider
#             if "_pipeline_meta" not in parsed:
#                 parsed["_pipeline_meta"] = {}
#             parsed["_pipeline_meta"].update({
#                 "edge_id":      event.get("edge_id",      ""),
#                 "scenario_id":  event.get("scenario_id",  ""),
#                 "t":            event.get("t",             0),
#                 "malicious":    event.get("malicious",     0),
#                 "attack_phase": event.get("attack_phase",  "benign"),
#                 "provider":     event.get("provider",      "AWS"),  # ← stamp input provider
#             })
#             return {"log": parsed}

#         try:
#             return _gen()
#         except Exception as e:
#             print(f"Generation failed after {max_retries} retries: {e} — using fallback")
#             return {"log": _fallback_log(event), "warning": str(e)[:100]}

#     @modal.method()
#     def generate(self, event: dict) -> dict:
#         return self._generate_single(event)

#     @modal.method()
#     def generate_batch(self, events: List[dict]) -> List[dict]:
#         results = []
#         for event in events:
#             try:
#                 results.append(self._generate_single(event))
#             except Exception as e:
#                 # Isolate — one bad event must not crash the whole batch
#                 results.append({"log": _fallback_log(event), "error": str(e)})
#         return results


# # ══════════════════════════════════════════════════════════════════════════════
# # FASTAPI WRAPPER
# # ══════════════════════════════════════════════════════════════════════════════

# @app.function(
#     gpu="T4",
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     container_idle_timeout=300,
#     allow_concurrent_inputs=5,
#     memory=16384,
# )
# @modal.asgi_app()
# def fastapi_app():
#     from fastapi import FastAPI, HTTPException, Security, Depends
#     from fastapi.responses import JSONResponse
#     from fastapi.security.api_key import APIKeyHeader
#     from pydantic import BaseModel, Field
#     from contextlib import asynccontextmanager

#     API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

#     def validate(api_key: str = Security(API_KEY_HEADER)):
#         expected = os.environ.get("STAGE_0B_API_KEY", "")
#         if not api_key or not _secrets.compare_digest(
#             hashlib.sha256(api_key.encode()).hexdigest(),
#             hashlib.sha256(expected.encode()).hexdigest(),
#         ):
#             raise HTTPException(status_code=403, detail="Invalid API key")
#         return api_key

#     @asynccontextmanager
#     async def lifespan(app):
#         print("Stage 0b FastAPI starting...")
#         yield

#     web       = FastAPI(title="Stage 0b — SIEM Log Generator", lifespan=lifespan)
#     generator = SIEMGenerator()

#     class GenerateRequest(BaseModel):
#         provider:      str
#         action:        str
#         entity_id:     str
#         target_id:     str = ""
#         source_ip:     str = ""
#         region:        str = "us-east-1"
#         cloud_account: str = ""
#         status:        str = "Success"
#         malicious:     int = 0
#         attack_phase:  str = "benign"
#         edge_id:       str = ""
#         scenario_id:   str = ""
#         t:             int = 0

#     class BatchGenerateRequest(BaseModel):
#         events:      List[GenerateRequest] = Field(..., max_length=2000)
#         scenario_id: str = ""

#     @web.get("/health")
#     async def health():
#         return {
#             "stage":           "0b",
#             "status":          "ok",
#             "model":           "sohomn/siem-log-generator-llama31-8b",
#             "batch_supported": True,
#             "max_batch_size":  50,
#         }

#     @web.get("/")
#     async def root():
#         return {
#             "service":   "Stage 0b — SIEM Log Generator",
#             "endpoints": {
#                 "POST /generate":       "Generate single log",
#                 "POST /generate_batch": "Generate batch of logs (max 50)",
#                 "GET  /health":         "Health check",
#             },
#         }

#     @web.post("/generate")
#     async def generate(req: GenerateRequest, _=Depends(validate)):
#         try:
#             result = generator.generate.remote(req.dict())
#             if "error" in result and not result.get("log"):
#                 raise HTTPException(status_code=500, detail=result["error"])
#             return result
#         except HTTPException:
#             raise
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

#     @web.post("/generate_batch")
#     async def generate_batch(req: BatchGenerateRequest, _=Depends(validate)):
#         try:
#             events_dict = [e.dict() for e in req.events]
#             # Override scenario_id on each event if provided at batch level
#             if req.scenario_id:
#                 for e in events_dict:
#                     if not e.get('scenario_id'):
#                         e['scenario_id'] = req.scenario_id
#             results  = generator.generate_batch.remote(events_dict)
#             failures = sum(1 for r in results if "error" in r)
#             return {
#                 "total":      len(results),
#                 "successful": len(results) - failures,
#                 "failed":     failures,
#                 "results":    results,
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Batch generation failed: {e}")

#     @web.exception_handler(Exception)
#     async def global_exc(request, exc):
#         return JSONResponse(status_code=500,
#                             content={"detail": f"Stage 0b error: {exc}"})

#     return web


"""
Stage 0b — SIEM Log Generator (LLaMA 3.1 8B + QLoRA)
Platform: Modal GPU T4
"""

import modal
import os
import json
import re
import torch
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import hashlib
import secrets as _secrets

# ============================================================
# MODAL IMAGE
# ============================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.43.0",
        "peft>=0.11.1",
        "bitsandbytes>=0.43.1",
        "accelerate>=0.30.0",
        "fastapi",
        "uvicorn",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece",
    )
)

app = modal.App("stage0b-siem-generator", image=image)

# ============================================================
# CONSTANTS
# ============================================================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_REPO = "sohomn/siem-log-generator-llama31-8b"

SYSTEM_PROMPT = (
    "You are a cloud security log renderer for a research pipeline. "
    "Given a structured security event, generate ONLY the corresponding "
    "cloud provider log as a valid JSON object. "
    "Output nothing except the JSON. No explanation. No markdown. "
    'The JSON must include a "_pipeline_meta" field with the original metadata.'
)


def repair_json(json_str: str) -> str:
    """Attempt to repair common JSON issues from LLM output"""
    # Remove markdown code blocks
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Add missing quotes around keys (simple cases)
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
    
    # Fix single quotes to double quotes
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
    # Remove comments (// or /* */)
    json_str = re.sub(r'//.*?($|\n)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix missing commas between objects in arrays (common LLM issue)
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # Fix missing commas between array items
    json_str = re.sub(r']\s*\[', '],[', json_str)
    
    # Remove control characters
    json_str = ''.join(ch for ch in json_str if ord(ch) >= 32 or ch in '\n\r\t')
    
    return json_str


def extract_json(text: str) -> Optional[str]:
    """Extract JSON from LLM response with multiple strategies"""
    
    # Strategy 1: Find first { and last }
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        candidate = text[start:end]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            # Try to repair
            repaired = repair_json(candidate)
            try:
                json.loads(repaired)
                return repaired
            except:
                pass
    
    # Strategy 2: Try to find any JSON-like structure
    import ast
    pattern = r'(\{.*\})'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            json.loads(match)
            return match
        except:
            repaired = repair_json(match)
            try:
                json.loads(repaired)
                return repaired
            except:
                continue
    
    return None


# ============================================================
# MODAL CLASS
# ============================================================
@app.cls(
    gpu="T4",
    memory=16384,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    scaledown_window=300,
)
class SIEMGenerator:
    @modal.enter()
    def load_model(self):
        """Load LLaMA model and LoRA adapter on container startup"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login as hf_login
        
        print(f"🔄 Loading Stage 0b SIEM Generator...")
        
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            hf_login(token=hf_token)
            print("  ✓ HF_TOKEN loaded")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")
        
        # Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        print(f"  Loading tokenizer from {BASE_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print(f"  Loading base model from {BASE_MODEL}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
        )
        
        # Load LoRA adapter
        print(f"  Loading LoRA adapter from {ADAPTER_REPO}...")
        self.model = PeftModel.from_pretrained(
            base_model, 
            ADAPTER_REPO, 
            is_trainable=False,
            token=hf_token if hf_token else None,
        )
        self.model.eval()
        
        print(f"✅ Stage 0b ready — LLaMA 3.1 8B + LoRA loaded")
    
    def _generate_safe(self, event: dict, max_retries: int = 2) -> dict:
        """Generate log with retries and improved JSON extraction"""
        import time
        
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(event)},
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=768
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.3,  # Lower temperature for more deterministic output
                        do_sample=False,  # Greedy decoding
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Extract JSON using improved method
                json_str = extract_json(response)
                
                if json_str is None:
                    raise ValueError(f"No valid JSON found in response: {response[:200]}")
                
                parsed = json.loads(json_str)
                
                # Ensure _pipeline_meta exists
                if "_pipeline_meta" not in parsed:
                    parsed["_pipeline_meta"] = {
                        "edge_id": event.get("edge_id", ""),
                        "scenario_id": event.get("scenario_id", ""),
                        "t": event.get("t", 0),
                        "malicious": int(event.get("malicious", 0)),
                        "attack_phase": event.get("attack_phase", "benign"),
                        "provider": event.get("provider", "AWS"),
                    }
                
                return {"log": parsed}
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"  ⚠️ Using fallback template for {event.get('scenario_id', 'unknown')}")
                    return self._generate_fallback(event)
                time.sleep(1)
        
        return self._generate_fallback(event)
    
    def _generate_fallback(self, event: dict) -> dict:
        """Generate fallback log (provider-native template)"""
        provider = event.get("provider", "AWS")
        
        if provider == "AWS":
            log = {
                "eventVersion": "1.08",
                "eventName": event.get("action", "Unknown"),
                "eventSource": f"{event.get('entity_type', 'iam')}.amazonaws.com",
                "eventTime": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
                "userIdentity": {"userName": event.get("entity_id", "unknown")},
                "sourceIPAddress": event.get("source_ip", "0.0.0.0"),
                "requestParameters": {"resourceId": event.get("target_id", "")},
            }
        elif provider == "Azure":
            log = {
                "operationName": event.get("action", "Unknown"),
                "caller": event.get("entity_id", "unknown"),
                "eventTimestamp": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
                "properties": {"resourceId": event.get("target_id", "")},
            }
        else:
            log = {
                "protoPayload": {
                    "methodName": event.get("action", "Unknown"),
                    "authenticationInfo": {"principalEmail": event.get("entity_id", "unknown")},
                },
                "timestamp": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
            }
        
        log["_pipeline_meta"] = {
            "edge_id": event.get("edge_id", ""),
            "scenario_id": event.get("scenario_id", ""),
            "t": event.get("t", 0),
            "malicious": int(event.get("malicious", 0)),
            "attack_phase": event.get("attack_phase", "benign"),
            "provider": provider,
            # "fallback": True,
        }
        
        return {"log": log}
    
    @modal.method()
    async def generate(self, event: dict) -> dict:
        return self._generate_safe(event)
    
    @modal.method()
    async def generate_batch(self, events: List[dict]) -> List[dict]:
        results = []
        for event in events:
            results.append(self._generate_safe(event))
        return results


# ============================================================
# FASTAPI WRAPPER
# ============================================================
@app.function(
    gpu="T4",
    memory=16384,
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
    from typing import List, Optional
    
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_0B_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key
    
    web = FastAPI(title="Stage 0b — SIEM Log Generator", version="2.0.0")
    generator = SIEMGenerator()
    
    class EventRequest(BaseModel):
        provider: str
        action: str
        entity_id: str
        target_id: str = ""
        source_ip: str = ""
        region: str = "us-east-1"
        cloud_account: str = ""
        status: str = "Success"
        malicious: int = 0
        attack_phase: str = "benign"
        edge_id: str = ""
        scenario_id: str = ""
        t: int = 0
        entity_type: str = "User"
    
    class BatchRequest(BaseModel):
        events: List[EventRequest]
        scenario_id: Optional[str] = ""
    
    @web.get("/health")
    async def health():
        return {"stage": "0b", "status": "ok", "model": f"{BASE_MODEL} + LoRA"}
    
    @web.post("/generate")
    async def generate(req: EventRequest, _=Depends(validate)):
        result = await generator.generate.remote.aio(req.dict())
        return result
    
    @web.post("/generate_batch")
    async def generate_batch(req: BatchRequest, _=Depends(validate)):
        events_dict = [e.dict() for e in req.events]
        results = await generator.generate_batch.remote.aio(events_dict)
        
        fallback_count = sum(1 for r in results 
                           if r.get('log', {}).get('_pipeline_meta', {}).get('fallback', False))
        
        return {
            "total": len(results),
            "successful": len(results),
            "failed": 0,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(results) if results else 0,
            "results": results,
        }
    
    return web