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


# """
# Stage 0b — SIEM Log Generator (LLaMA 3.1 8B + QLoRA)
# Platform: Modal GPU T4
# """

# import modal
# import os
# import json
# import re
# import torch
# from typing import List, Dict, Optional
# from pydantic import BaseModel
# from fastapi import FastAPI, HTTPException, Security, Depends
# from fastapi.security.api_key import APIKeyHeader
# import hashlib
# import secrets as _secrets

# # ============================================================
# # MODAL IMAGE
# # ============================================================
# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install(
#         "torch>=2.0.0",
#         "transformers>=4.43.0",
#         "peft>=0.11.1",
#         "bitsandbytes>=0.43.1",
#         "accelerate>=0.30.0",
#         "fastapi",
#         "uvicorn",
#         "pydantic>=2.5.0",
#         "huggingface_hub>=0.23.0",
#         "sentencepiece",
#     )
# )

# app = modal.App("stage0b-siem-generator", image=image)

# # ============================================================
# # CONSTANTS
# # ============================================================
# BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# ADAPTER_REPO = "sohomn/siem-log-generator-llama31-8b"

# SYSTEM_PROMPT = (
#     "You are a cloud security log renderer for a research pipeline. "
#     "Given a structured security event, generate ONLY the corresponding "
#     "cloud provider log as a valid JSON object. "
#     "Output nothing except the JSON. No explanation. No markdown. "
#     'The JSON must include a "_pipeline_meta" field with the original metadata.'
# )


# def repair_json(json_str: str) -> str:
#     """Attempt to repair common JSON issues from LLM output"""
#     # Remove markdown code blocks
#     json_str = re.sub(r'```json\s*', '', json_str)
#     json_str = re.sub(r'```\s*', '', json_str)
    
#     # Remove trailing commas before closing braces/brackets
#     json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
#     # Add missing quotes around keys (simple cases)
#     json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
    
#     # Fix single quotes to double quotes
#     json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
#     # Remove comments (// or /* */)
#     json_str = re.sub(r'//.*?($|\n)', '', json_str)
#     json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
#     # Fix missing commas between objects in arrays (common LLM issue)
#     json_str = re.sub(r'}\s*{', '},{', json_str)
    
#     # Fix missing commas between array items
#     json_str = re.sub(r']\s*\[', '],[', json_str)
    
#     # Remove control characters
#     json_str = ''.join(ch for ch in json_str if ord(ch) >= 32 or ch in '\n\r\t')
    
#     return json_str


# def extract_json(text: str) -> Optional[str]:
#     """Extract JSON from LLM response with multiple strategies"""
    
#     # Strategy 1: Find first { and last }
#     start = text.find('{')
#     end = text.rfind('}') + 1
#     if start >= 0 and end > start:
#         candidate = text[start:end]
#         try:
#             json.loads(candidate)
#             return candidate
#         except json.JSONDecodeError:
#             # Try to repair
#             repaired = repair_json(candidate)
#             try:
#                 json.loads(repaired)
#                 return repaired
#             except:
#                 pass
    
#     # Strategy 2: Try to find any JSON-like structure
#     import ast
#     pattern = r'(\{.*\})'
#     matches = re.findall(pattern, text, re.DOTALL)
#     for match in matches:
#         try:
#             json.loads(match)
#             return match
#         except:
#             repaired = repair_json(match)
#             try:
#                 json.loads(repaired)
#                 return repaired
#             except:
#                 continue
    
#     return None


# # ============================================================
# # MODAL CLASS
# # ============================================================
# @app.cls(
#     gpu="T4",
#     memory=16384,
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     scaledown_window=300,
# )
# class SIEMGenerator:
#     @modal.enter()
#     def load_model(self):
#         """Load LLaMA model and LoRA adapter on container startup"""
#         import torch
#         from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#         from peft import PeftModel
#         from huggingface_hub import login as hf_login
        
#         print(f"🔄 Loading Stage 0b SIEM Generator...")
        
#         hf_token = os.environ.get("HF_TOKEN", "")
#         if hf_token:
#             hf_login(token=hf_token)
#             print("  ✓ HF_TOKEN loaded")
        
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"  Device: {self.device}")
        
#         # Quantization config (4-bit)
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )
        
#         # Load tokenizer
#         print(f"  Loading tokenizer from {BASE_MODEL}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             BASE_MODEL,
#             trust_remote_code=True,
#             token=hf_token if hf_token else None,
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Load base model
#         print(f"  Loading base model from {BASE_MODEL}...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL,
#             quantization_config=bnb_config,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             token=hf_token if hf_token else None,
#         )
        
#         # Load LoRA adapter
#         print(f"  Loading LoRA adapter from {ADAPTER_REPO}...")
#         self.model = PeftModel.from_pretrained(
#             base_model, 
#             ADAPTER_REPO, 
#             is_trainable=False,
#             token=hf_token if hf_token else None,
#         )
#         self.model.eval()
        
#         print(f"✅ Stage 0b ready — LLaMA 3.1 8B + LoRA loaded")
    
#     def _generate_safe(self, event: dict, max_retries: int = 2) -> dict:
#         """Generate log with retries and improved JSON extraction"""
#         import time
        
#         for attempt in range(max_retries):
#             try:
#                 messages = [
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": json.dumps(event)},
#                 ]
                
#                 text = self.tokenizer.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=True
#                 )
                
#                 inputs = self.tokenizer(
#                     text, return_tensors="pt", truncation=True, max_length=768
#                 ).to(self.model.device)
                
#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=512,
#                         temperature=0.3,  # Lower temperature for more deterministic output
#                         do_sample=False,  # Greedy decoding
#                         pad_token_id=self.tokenizer.eos_token_id,
#                     )
                
#                 response = self.tokenizer.decode(
#                     outputs[0][inputs["input_ids"].shape[1]:], 
#                     skip_special_tokens=True
#                 ).strip()
                
#                 # Extract JSON using improved method
#                 json_str = extract_json(response)
                
#                 if json_str is None:
#                     raise ValueError(f"No valid JSON found in response: {response[:200]}")
                
#                 parsed = json.loads(json_str)
                
#                 # Ensure _pipeline_meta exists
#                 if "_pipeline_meta" not in parsed:
#                     parsed["_pipeline_meta"] = {
#                         "edge_id": event.get("edge_id", ""),
#                         "scenario_id": event.get("scenario_id", ""),
#                         "t": event.get("t", 0),
#                         "malicious": int(event.get("malicious", 0)),
#                         "attack_phase": event.get("attack_phase", "benign"),
#                         "provider": event.get("provider", "AWS"),
#                     }
                
#                 return {"log": parsed}
                
#             except Exception as e:
#                 print(f"  Attempt {attempt + 1} failed: {e}")
#                 if attempt == max_retries - 1:
#                     print(f"  ⚠️ Using fallback template for {event.get('scenario_id', 'unknown')}")
#                     return self._generate_fallback(event)
#                 time.sleep(1)
        
#         return self._generate_fallback(event)
    
#     def _generate_fallback(self, event: dict) -> dict:
#         """Generate fallback log (provider-native template)"""
#         provider = event.get("provider", "AWS")
        
#         if provider == "AWS":
#             log = {
#                 "eventVersion": "1.08",
#                 "eventName": event.get("action", "Unknown"),
#                 "eventSource": f"{event.get('entity_type', 'iam')}.amazonaws.com",
#                 "eventTime": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
#                 "userIdentity": {"userName": event.get("entity_id", "unknown")},
#                 "sourceIPAddress": event.get("source_ip", "0.0.0.0"),
#                 "requestParameters": {"resourceId": event.get("target_id", "")},
#             }
#         elif provider == "Azure":
#             log = {
#                 "operationName": event.get("action", "Unknown"),
#                 "caller": event.get("entity_id", "unknown"),
#                 "eventTimestamp": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
#                 "properties": {"resourceId": event.get("target_id", "")},
#             }
#         else:
#             log = {
#                 "protoPayload": {
#                     "methodName": event.get("action", "Unknown"),
#                     "authenticationInfo": {"principalEmail": event.get("entity_id", "unknown")},
#                 },
#                 "timestamp": f"2025-01-15T{event.get('t', 0):02d}:00:00Z",
#             }
        
#         log["_pipeline_meta"] = {
#             "edge_id": event.get("edge_id", ""),
#             "scenario_id": event.get("scenario_id", ""),
#             "t": event.get("t", 0),
#             "malicious": int(event.get("malicious", 0)),
#             "attack_phase": event.get("attack_phase", "benign"),
#             "provider": provider,
#             # "fallback": True,
#         }
        
#         return {"log": log}
    
#     @modal.method()
#     async def generate(self, event: dict) -> dict:
#         return self._generate_safe(event)
    
#     @modal.method()
#     async def generate_batch(self, events: List[dict]) -> List[dict]:
#         results = []
#         for event in events:
#             results.append(self._generate_safe(event))
#         return results


# # ============================================================
# # FASTAPI WRAPPER
# # ============================================================
# @app.function(
#     gpu="T4",
#     memory=16384,
#     secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
#     scaledown_window=300,
# )
# @modal.asgi_app()
# def fastapi_app():
#     import os
#     import hashlib
#     import secrets as _secrets
#     from fastapi import FastAPI, HTTPException, Security, Depends
#     from fastapi.security.api_key import APIKeyHeader
#     from pydantic import BaseModel
#     from typing import List, Optional
    
#     API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
    
#     def validate(api_key: str = Security(API_KEY_HEADER)):
#         expected = os.environ.get("STAGE_0B_API_KEY", "")
#         if not api_key or not _secrets.compare_digest(
#             hashlib.sha256(api_key.encode()).hexdigest(),
#             hashlib.sha256(expected.encode()).hexdigest(),
#         ):
#             raise HTTPException(status_code=403, detail="Invalid API key")
#         return api_key
    
#     web = FastAPI(title="Stage 0b — SIEM Log Generator", version="2.0.0")
#     generator = SIEMGenerator()
    
#     class EventRequest(BaseModel):
#         provider: str
#         action: str
#         entity_id: str
#         target_id: str = ""
#         source_ip: str = ""
#         region: str = "us-east-1"
#         cloud_account: str = ""
#         status: str = "Success"
#         malicious: int = 0
#         attack_phase: str = "benign"
#         edge_id: str = ""
#         scenario_id: str = ""
#         t: int = 0
#         entity_type: str = "User"
    
#     class BatchRequest(BaseModel):
#         events: List[EventRequest]
#         scenario_id: Optional[str] = ""
    
#     @web.get("/health")
#     async def health():
#         return {"stage": "0b", "status": "ok", "model": f"{BASE_MODEL} + LoRA"}
    
#     @web.post("/generate")
#     async def generate(req: EventRequest, _=Depends(validate)):
#         result = await generator.generate.remote.aio(req.dict())
#         return result
    
#     @web.post("/generate_batch")
#     async def generate_batch(req: BatchRequest, _=Depends(validate)):
#         events_dict = [e.dict() for e in req.events]
#         results = await generator.generate_batch.remote.aio(events_dict)
        
#         fallback_count = sum(1 for r in results 
#                            if r.get('log', {}).get('_pipeline_meta', {}).get('fallback', False))
        
#         return {
#             "total": len(results),
#             "successful": len(results),
#             "failed": 0,
#             "fallback_count": fallback_count,
#             "fallback_rate": fallback_count / len(results) if results else 0,
#             "results": results,
#         }
    
#     return web


# """
# Stage 0b — SIEM Log Generator (Groq API)
# =========================================
# Replaces: LLaMA fine-tuning + Modal GPU deployment
# Platform:  Render free tier (CPU) or any machine with internet
# Model:     llama-3.1-8b-instant via Groq (free tier: 6000 req/min)

# Usage:
#     pip install groq fastapi uvicorn pydantic pandas pyarrow
#     export GROQ_API_KEY=your_key
#     export STAGE_0B_API_KEY=your_api_key   # for the /generate endpoints
#     uvicorn app:web --host 0.0.0.0 --port 8000

#     # OR run batch generation directly:
#     python app.py --batch --input structured_events.parquet --output ./outputs
# """

# import os, json, re, time, hashlib, secrets as _secrets, uuid, argparse
# from datetime import datetime, timezone, timedelta
# from typing import Optional
# import hashlib

# from fastapi import FastAPI, HTTPException, Security, Depends
# from fastapi.security.api_key import APIKeyHeader
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, field_validator
# from groq import Groq

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────
# GROQ_MODEL    = "llama-3.1-8b-instant"   # free, fast, same base model as your adapter
# GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
# API_KEY       = os.environ.get("STAGE_0B_API_KEY", "")

# # Groq free tier: 6000 RPM / 500k tokens per day on llama-3.1-8b-instant
# # Each request ≈ 600 prompt tokens + 400 completion = ~1000 tokens
# # 500k / 1000 = 500 events per day on free tier
# # Paid tier: no daily limit, just RPM
# RATE_LIMIT_RPM = 30     # conservative — stay well under Groq's 6000 RPM
# REQUEST_DELAY  = 60 / RATE_LIMIT_RPM   # seconds between requests

# # ─────────────────────────────────────────────────────────────────────────────
# # PROVIDER-SPECIFIC SYSTEM PROMPTS (from your notebook Cell 5)
# # ─────────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPTS = {
#     "AWS": (
#         "You are an AWS CloudTrail log renderer for a security research pipeline. "
#         "Given a structured security event as JSON, generate ONLY the corresponding "
#         "AWS CloudTrail log entry as a single valid JSON object matching the CloudTrail 1.08 schema.\n"
#         "Required top-level fields: eventVersion, eventTime (ISO8601), eventSource "
#         "(service.amazonaws.com format), eventName, awsRegion, sourceIPAddress, userAgent, "
#         "requestParameters, responseElements, errorCode (null if success), userIdentity "
#         "(type, principalId, arn, accountId, userName), requestID, eventID, "
#         "readOnly (bool), eventType (AwsApiCall), recipientAccountId.\n"
#         "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
#         "malicious, attack_phase, provider exactly from the input.\n"
#         "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
#         "Start your response with { and end with }."
#     ),
#     "Azure": (
#         "You are an Azure Activity Log renderer for a security research pipeline. "
#         "Given a structured security event as JSON, generate ONLY the corresponding "
#         "Azure Activity Log entry as a single valid JSON object matching the Azure Monitor schema.\n"
#         "Required top-level fields: time (ISO8601), resourceId (full ARM path), "
#         "operationName (Microsoft.Provider/resource/action format), operationVersion, "
#         "category (Administrative), resultType (Succeeded/Failed), resultSignature (HTTP code), "
#         "durationMs (integer), callerIpAddress, correlationId (GUID), "
#         "identity (authorization object + claims with name and upn), "
#         "level (Information/Warning), location, "
#         "properties (statusCode, serviceRequestId, resourceId).\n"
#         "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
#         "malicious, attack_phase, provider exactly from the input.\n"
#         "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
#         "Start your response with { and end with }."
#     ),
#     "GCP": (
#         "You are a GCP Cloud Audit Log renderer for a security research pipeline. "
#         "Given a structured security event as JSON, generate ONLY the corresponding "
#         "GCP Cloud Audit Log entry as a single valid JSON object matching the Cloud Audit Log schema.\n"
#         "Required top-level fields: logName (projects/PROJECT/logs/cloudaudit.googleapis.com%2Factivity), "
#         "resource (type + labels with project_id, zone, instance_id), "
#         "timestamp (RFC3339), severity (NOTICE), "
#         "protoPayload (@type as type.googleapis.com/google.cloud.audit.AuditLog, "
#         "status object, authenticationInfo with principalEmail, "
#         "requestMetadata with callerIp and callerSuppliedUserAgent, "
#         "serviceName, methodName, authorizationInfo array, resourceName), "
#         "insertId, receiveTimestamp.\n"
#         "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
#         "malicious, attack_phase, provider exactly from the input.\n"
#         "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
#         "Start your response with { and end with }."
#     ),
# }

# # Input fields sent to the model (exactly what Stage 0a produces)
# INPUT_FIELDS = [
#     "provider", "action", "entity_id", "entity_type", "target_id",
#     "source_ip", "region", "cloud_account", "status",
#     "malicious", "attack_phase", "edge_id", "scenario_id", "t",
# ]

# # ─────────────────────────────────────────────────────────────────────────────
# # JSON UTILITIES (identical to your notebook)
# # ─────────────────────────────────────────────────────────────────────────────
# def repair_json(text: str) -> str:
#     text = re.sub(r"```json\s*", "", text)
#     text = re.sub(r"```\s*",     "", text)
#     text = re.sub(r",(\s*[}\]])", r"\1", text)
#     text = re.sub(r"'([^']*)':", r'"\1":', text)
#     text = re.sub(r"//.*?($|\n)", "", text, flags=re.MULTILINE)
#     text = re.sub(r"/\*.*?\*/",   "", text, flags=re.DOTALL)
#     text = re.sub(r"}\s*{", "},{", text)
#     text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")
#     return text


# def extract_json(text: str) -> Optional[dict]:
#     # Strategy 1: outer {…}
#     s, e = text.find("{"), text.rfind("}") + 1
#     if s >= 0 and e > s:
#         candidate = text[s:e]
#         try:
#             return json.loads(candidate)
#         except json.JSONDecodeError:
#             try:
#                 return json.loads(repair_json(candidate))
#             except Exception:
#                 pass
#     # Strategy 2: greedy regex
#     for m in re.findall(r"(\{.*\})", text, re.DOTALL):
#         try:
#             return json.loads(m)
#         except Exception:
#             try:
#                 return json.loads(repair_json(m))
#             except Exception:
#                 continue
#     return None


# # ─────────────────────────────────────────────────────────────────────────────
# # DETERMINISTIC FALLBACK BUILDERS
# # Used when Groq returns unparseable output after retries.
# # ─────────────────────────────────────────────────────────────────────────────
# def _sid(seed: str, prefix: str = "") -> str:
#     h = hashlib.md5(seed.encode()).hexdigest()
#     return f"{prefix}{h[:12]}" if prefix else h[:16]

# def _ts(t: int, scenario_id: str) -> str:
#     base = datetime(2025, 1, 15, tzinfo=timezone.utc)
#     ts   = base + timedelta(hours=(hash(scenario_id) % 24 + t) % 24,
#                             minutes=hash(scenario_id + str(t)) % 60)
#     return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

# def _guid(seed: str) -> str:
#     return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))

# def _fallback(ev: dict) -> dict:
#     p   = ev.get("provider", "AWS")
#     ts  = _ts(ev.get("t", 0), ev.get("scenario_id", ""))
#     uid = ev.get("entity_id", "unknown")
#     src = ev.get("source_ip", "0.0.0.0")
#     act = ev.get("action",    "UnknownAction")
#     tgt = ev.get("target_id", "")
#     acct= ev.get("cloud_account", "123456789012")
#     rgn = ev.get("region",    "us-east-1")
#     st  = ev.get("status",    "Success")

#     if p == "AWS":
#         svc_map = {"AssumeRole": "sts", "RunInstances": "ec2",
#                    "GetObject": "s3", "InvokeFunction": "lambda"}
#         svc = svc_map.get(act, "iam")
#         log = {
#             "eventVersion": "1.08", "eventTime": ts,
#             "eventSource": f"{svc}.amazonaws.com", "eventName": act,
#             "awsRegion": rgn, "sourceIPAddress": src,
#             "userAgent": "aws-cli/2.15.0",
#             "userIdentity": {"type": "IAMUser", "userName": uid,
#                              "arn": f"arn:aws:iam::{acct}:user/{uid}", "accountId": acct},
#             "requestParameters": {"resourceId": tgt} if tgt else None,
#             "responseElements": None,
#             "errorCode": None if st == "Success" else "AccessDenied",
#             "requestID": _guid(f"{uid}{ts}req"), "eventID": _guid(f"{uid}{ts}evt"),
#             "readOnly": act.startswith("Describe") or act.startswith("Get"),
#             "eventType": "AwsApiCall", "recipientAccountId": acct,
#         }
#     elif p == "Azure":
#         op = act if "/" in act else f"Microsoft.Compute/virtualMachines/{act}"
#         log = {
#             "time": ts,
#             "resourceId": f"/subscriptions/{acct}/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/{tgt}",
#             "operationName": op, "operationVersion": "2023-07-01",
#             "category": "Administrative",
#             "resultType": "Succeeded" if st == "Success" else "Failed",
#             "resultSignature": "200" if st == "Success" else "403",
#             "durationMs": 312, "callerIpAddress": src,
#             "correlationId": _guid(f"{uid}{ts}corr"),
#             "identity": {"claims": {"name": uid, "upn": f"{uid}@corp.onmicrosoft.com"}},
#             "level": "Information", "location": rgn,
#             "properties": {"statusCode": "OK" if st == "Success" else "Forbidden",
#                            "serviceRequestId": _guid(f"{uid}{ts}svc")},
#         }
#     else:  # GCP
#         method = act if "." in act else f"compute.instances.{act.lower()}"
#         log = {
#             "logName": f"projects/{acct}/logs/cloudaudit.googleapis.com%2Factivity",
#             "resource": {"type": "gce_instance",
#                          "labels": {"project_id": acct, "zone": f"{rgn}-a"}},
#             "timestamp": ts, "severity": "NOTICE",
#             "protoPayload": {
#                 "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
#                 "status": {} if st == "Success" else {"code": 7, "message": "PERMISSION_DENIED"},
#                 "authenticationInfo": {"principalEmail": f"{uid}@{acct}.iam.gserviceaccount.com"},
#                 "requestMetadata": {"callerIp": src},
#                 "serviceName": method.split(".")[0] + ".googleapis.com",
#                 "methodName": method,
#                 "resourceName": f"projects/{acct}/zones/{rgn}-a/instances/{tgt}",
#             },
#             "insertId": _sid(f"{uid}{ts}"),
#             "receiveTimestamp": ts,
#         }

#     log["_pipeline_meta"] = _make_meta(ev)
#     return log


# def _make_meta(ev: dict) -> dict:
#     return {
#         "edge_id":      ev.get("edge_id", ""),
#         "scenario_id":  ev.get("scenario_id", ""),
#         "t":            int(ev.get("t", 0)),
#         "malicious":    int(ev.get("malicious", 0)),
#         "attack_phase": ev.get("attack_phase", "benign"),
#         "provider":     ev.get("provider", "AWS"),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # GROQ GENERATOR
# # ─────────────────────────────────────────────────────────────────────────────
# _groq_client: Optional[Groq] = None

# def get_client() -> Groq:
#     global _groq_client
#     if _groq_client is None:
#         key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
#         if not key:
#             raise RuntimeError("GROQ_API_KEY not set")
#         _groq_client = Groq(api_key=key)
#     return _groq_client


# def generate_one(ev: dict, max_retries: int = 3) -> dict:
#     """
#     Call Groq to generate a provider-native log for one structured event.
#     Returns {"log": dict, "fallback": bool}.
#     """
#     client   = get_client()
#     provider = ev.get("provider", "AWS")
#     system   = SYSTEM_PROMPTS.get(provider, SYSTEM_PROMPTS["AWS"])
#     payload  = {k: ev.get(k, "") for k in INPUT_FIELDS if k in ev}
#     user_msg = json.dumps(payload, default=str)

#     last_err = None
#     for attempt in range(max_retries):
#         try:
#             resp = client.chat.completions.create(
#                 model=GROQ_MODEL,
#                 messages=[
#                     {"role": "system",  "content": system},
#                     {"role": "user",    "content": user_msg},
#                 ],
#                 temperature=0.1,     # near-deterministic — schema task not creative
#                 max_tokens=800,
#                 stop=None,
#             )
#             raw    = resp.choices[0].message.content.strip()
#             parsed = extract_json(raw)

#             if parsed is None:
#                 raise ValueError(f"JSON extraction failed. Raw[:150]: {raw[:150]}")

#             # Always overwrite _pipeline_meta — never trust the model for labels
#             parsed["_pipeline_meta"] = _make_meta(ev)
#             return {"log": parsed, "fallback": False}

#         except Exception as exc:
#             last_err = exc
#             # Groq rate limit → back off
#             if "rate_limit" in str(exc).lower() or "429" in str(exc):
#                 wait = 10 * (attempt + 1)
#                 print(f"  Rate limit hit, waiting {wait}s …")
#                 time.sleep(wait)
#             else:
#                 time.sleep(1)

#     # All retries failed → deterministic fallback
#     print(f"  ⚠️  Groq failed after {max_retries} attempts: {last_err}. Using fallback.")
#     return {"log": _fallback(ev), "fallback": True}


# # ─────────────────────────────────────────────────────────────────────────────
# # FASTAPI APP  (for Render deployment)
# # ─────────────────────────────────────────────────────────────────────────────
# API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# def _validate(api_key: str = Security(API_KEY_HEADER)) -> str:
#     expected = API_KEY or os.environ.get("STAGE_0B_API_KEY", "")
#     if not expected:
#         return api_key or ""   # no key configured → open (dev mode)
#     if not api_key or not _secrets.compare_digest(
#         hashlib.sha256(api_key.encode()).hexdigest(),
#         hashlib.sha256(expected.encode()).hexdigest(),
#     ):
#         raise HTTPException(status_code=403, detail="Invalid or missing API key")
#     return api_key


# class EventRequest(BaseModel):
#     provider:      str = Field("AWS")
#     action:        str = Field(...)
#     entity_id:     str = Field(...)
#     entity_type:   str = Field("User")
#     target_id:     str = Field("")
#     source_ip:     str = Field("0.0.0.0")
#     region:        str = Field("us-east-1")
#     cloud_account: str = Field("")
#     status:        str = Field("Success")
#     malicious:     int = Field(0)
#     attack_phase:  str = Field("benign")
#     edge_id:       str = Field("")
#     scenario_id:   str = Field("")
#     t:             int = Field(0)

#     @field_validator("provider")
#     @classmethod
#     def valid_provider(cls, v):
#         if v not in ("AWS", "Azure", "GCP"):
#             raise ValueError("provider must be AWS, Azure, or GCP")
#         return v


# class BatchRequest(BaseModel):
#     events: list[EventRequest]


# web = FastAPI(
#     title="Trinetra — Stage 0b SIEM Log Generator (Groq)",
#     version="4.0.0",
#     description="Converts Stage 0a structured events into provider-native cloud audit logs via Groq LLaMA-3.1-8B.",
# )
# web.add_middleware(CORSMiddleware, allow_origins=["*"],
#                    allow_methods=["GET", "POST"], allow_headers=["*"])


# @web.get("/health")
# def health():
#     return {"stage": "0b", "backend": "groq",
#             "model": GROQ_MODEL, "status": "ok", "version": "4.0.0"}


# @web.post("/generate")
# def generate(req: EventRequest, _: str = Depends(_validate)):
#     return generate_one(req.model_dump())


# @web.post("/generate_batch")
# def generate_batch(req: BatchRequest, _: str = Depends(_validate)):
#     results, fallbacks = [], 0
#     for ev in req.events:
#         r = generate_one(ev.model_dump())
#         results.append(r)
#         if r["fallback"]:
#             fallbacks += 1
#         time.sleep(REQUEST_DELAY)   # respect rate limit between calls
#     n = len(results)
#     return {
#         "total": n, "successful": n,
#         "fallback_count": fallbacks,
#         "fallback_rate":  round(fallbacks / n, 3) if n else 0,
#         "results": results,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # BATCH GENERATION SCRIPT  (run directly: python app.py --batch …)
# # Reads structured_events.parquet from Stage 0a, writes 4 parquets for Stage 1
# # ─────────────────────────────────────────────────────────────────────────────
# def run_batch(input_path: str, output_dir: str, checkpoint_every: int = 500):
#     import pandas as pd
#     from pathlib import Path
#     from tqdm import tqdm

#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     combined_path   = os.path.join(output_dir, "stage0b_combined.parquet")
#     checkpoint_path = os.path.join(output_dir, "_checkpoint.parquet")

#     df = pd.read_parquet(input_path)
#     print(f"Loaded {len(df):,} events from {input_path}")
#     print(f"Provider counts:\n{df['provider'].value_counts().to_string()}")

#     # Resume from checkpoint if it exists
#     already_done = set()
#     done_rows    = []
#     if os.path.exists(checkpoint_path):
#         ckpt_df = pd.read_parquet(checkpoint_path)
#         already_done = set(zip(ckpt_df["scenario_id"], ckpt_df["t"], ckpt_df["edge_id"]))
#         done_rows    = ckpt_df.to_dict("records")
#         print(f"Resuming: {len(done_rows):,} already processed, {len(df)-len(done_rows):,} remaining")

#     events     = df.to_dict("records")
#     results    = list(done_rows)
#     fallbacks  = 0

#     for ev in tqdm(events):
#         key = (str(ev.get("scenario_id","")), str(ev.get("t","")), str(ev.get("edge_id","")))
#         if key in already_done:
#             continue

#         r = generate_one(ev)
#         if r["fallback"]:
#             fallbacks += 1

#         meta = r["log"].get("_pipeline_meta", _make_meta(ev))
#         results.append({
#             "raw_log":      json.dumps(r["log"], separators=(",",":"), default=str),
#             "edge_id":      meta.get("edge_id", ""),
#             "scenario_id":  meta.get("scenario_id", ""),
#             "t":            meta.get("t", 0),
#             "malicious":    meta.get("malicious", 0),
#             "attack_phase": meta.get("attack_phase", "benign"),
#             "provider":     meta.get("provider", "AWS"),
#             "_fallback":    r["fallback"],
#         })
#         already_done.add(key)
#         time.sleep(REQUEST_DELAY)

#         # Save checkpoint every N rows
#         if len(results) % checkpoint_every == 0:
#             pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
#             print(f"  Checkpoint saved ({len(results):,} done, {fallbacks} fallbacks)")

#     # Final save
#     out_df = pd.DataFrame(results)
#     out_df.to_parquet(combined_path, index=False)

#     for prov in ("AWS", "Azure", "GCP"):
#         sub = out_df[out_df["provider"] == prov]
#         sub.to_parquet(os.path.join(output_dir, f"stage0b_{prov.lower()}.parquet"), index=False)
#         print(f"  {prov}: {len(sub):,} rows")

#     print(f"\nDone. {len(out_df):,} total logs, {fallbacks} fallbacks ({100*fallbacks/len(out_df):.1f}%)")
#     print(f"Outputs in: {output_dir}")

#     # Clean up checkpoint
#     if os.path.exists(checkpoint_path):
#         os.remove(checkpoint_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch",   action="store_true", help="Run batch generation")
#     parser.add_argument("--input",   default="structured_events.parquet")
#     parser.add_argument("--output",  default="./stage0b_outputs")
#     parser.add_argument("--checkpoint-every", type=int, default=500)
#     parser.add_argument("--serve",   action="store_true", help="Run FastAPI server")
#     parser.add_argument("--port",    type=int, default=8000)
#     args = parser.parse_args()

#     if args.batch:
#         run_batch(args.input, args.output, args.checkpoint_every)
#     elif args.serve:
#         import uvicorn
#         uvicorn.run("app:web", host="0.0.0.0", port=args.port, reload=False)
#     else:
#         parser.print_help()



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
Stage 0b — SIEM Log Generator (Groq API)
=========================================
Replaces: LLaMA fine-tuning + Modal GPU deployment
Platform:  Render free tier (CPU) or any machine with internet
Model:     llama-3.1-8b-instant via Groq (free tier: 6000 req/min)

Usage:
    pip install groq fastapi uvicorn pydantic pandas pyarrow
    export GROQ_API_KEY=your_key
    export STAGE_0B_API_KEY=your_api_key   # for the /generate endpoints
    uvicorn app:web --host 0.0.0.0 --port 8000

    # OR run batch generation directly:
    python app.py --batch --input structured_events.parquet --output ./outputs
"""

import os, json, re, time, hashlib, secrets as _secrets, uuid, argparse
from datetime import datetime, timezone, timedelta
from typing import Optional
import hashlib

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.1-8b-instant"   # free, fast, same base model as your adapter
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
API_KEY       = os.environ.get("STAGE_0B_API_KEY", "")

# Groq free tier: 6000 RPM / 500k tokens per day on llama-3.1-8b-instant
# Each request ≈ 600 prompt tokens + 400 completion = ~1000 tokens
# 500k / 1000 = 500 events per day on free tier
# Paid tier: no daily limit, just RPM
RATE_LIMIT_RPM = 30     # conservative — stay well under Groq's 6000 RPM
REQUEST_DELAY  = 60 / RATE_LIMIT_RPM   # seconds between requests

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER-SPECIFIC SYSTEM PROMPTS (from your notebook Cell 5)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "AWS": (
        "You are an AWS CloudTrail log renderer for a security research pipeline. "
        "Given a structured security event as JSON, generate ONLY the corresponding "
        "AWS CloudTrail log entry as a single valid JSON object matching the CloudTrail 1.08 schema.\n"
        "Required top-level fields: eventVersion, eventTime (ISO8601), eventSource "
        "(service.amazonaws.com format), eventName, awsRegion, sourceIPAddress, userAgent, "
        "requestParameters, responseElements, errorCode (null if success), userIdentity "
        "(type, principalId, arn, accountId, userName), requestID, eventID, "
        "readOnly (bool), eventType (AwsApiCall), recipientAccountId.\n"
        "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
        "malicious, attack_phase, provider exactly from the input.\n"
        "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
        "Start your response with { and end with }."
    ),
    "Azure": (
        "You are an Azure Activity Log renderer for a security research pipeline. "
        "Given a structured security event as JSON, generate ONLY the corresponding "
        "Azure Activity Log entry as a single valid JSON object matching the Azure Monitor schema.\n"
        "Required top-level fields: time (ISO8601), resourceId (full ARM path), "
        "operationName (Microsoft.Provider/resource/action format), operationVersion, "
        "category (Administrative), resultType (Succeeded/Failed), resultSignature (HTTP code), "
        "durationMs (integer), callerIpAddress, correlationId (GUID), "
        "identity (authorization object + claims with name and upn), "
        "level (Information/Warning), location, "
        "properties (statusCode, serviceRequestId, resourceId).\n"
        "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
        "malicious, attack_phase, provider exactly from the input.\n"
        "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
        "Start your response with { and end with }."
    ),
    "GCP": (
        "You are a GCP Cloud Audit Log renderer for a security research pipeline. "
        "Given a structured security event as JSON, generate ONLY the corresponding "
        "GCP Cloud Audit Log entry as a single valid JSON object matching the Cloud Audit Log schema.\n"
        "Required top-level fields: logName (projects/PROJECT/logs/cloudaudit.googleapis.com%2Factivity), "
        "resource (type + labels with project_id, zone, instance_id), "
        "timestamp (RFC3339), severity (NOTICE), "
        "protoPayload (@type as type.googleapis.com/google.cloud.audit.AuditLog, "
        "status object, authenticationInfo with principalEmail, "
        "requestMetadata with callerIp and callerSuppliedUserAgent, "
        "serviceName, methodName, authorizationInfo array, resourceName), "
        "insertId, receiveTimestamp.\n"
        "Also include a \"_pipeline_meta\" field copying edge_id, scenario_id, t, "
        "malicious, attack_phase, provider exactly from the input.\n"
        "Output ONLY the JSON object. No markdown. No explanation. No extra text. "
        "Start your response with { and end with }."
    ),
}

# Input fields sent to the model (exactly what Stage 0a produces)
INPUT_FIELDS = [
    "provider", "action", "entity_id", "entity_type", "target_id",
    "source_ip", "region", "cloud_account", "status",
    "malicious", "attack_phase", "edge_id", "scenario_id", "t",
]

# ─────────────────────────────────────────────────────────────────────────────
# JSON UTILITIES (identical to your notebook)
# ─────────────────────────────────────────────────────────────────────────────
def repair_json(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*",     "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r"//.*?($|\n)", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/",   "", text, flags=re.DOTALL)
    text = re.sub(r"}\s*{", "},{", text)
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")
    return text


def extract_json(text: str) -> Optional[dict]:
    # Strategy 1: outer {…}
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        candidate = text[s:e]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(candidate))
            except Exception:
                pass
    # Strategy 2: greedy regex
    for m in re.findall(r"(\{.*\})", text, re.DOTALL):
        try:
            return json.loads(m)
        except Exception:
            try:
                return json.loads(repair_json(m))
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK BUILDERS
# Used when Groq returns unparseable output after retries.
# ─────────────────────────────────────────────────────────────────────────────
def _sid(seed: str, prefix: str = "") -> str:
    h = hashlib.md5(seed.encode()).hexdigest()
    return f"{prefix}{h[:12]}" if prefix else h[:16]

def _ts(t: int, scenario_id: str) -> str:
    base = datetime(2025, 1, 15, tzinfo=timezone.utc)
    ts   = base + timedelta(hours=(hash(scenario_id) % 24 + t) % 24,
                            minutes=hash(scenario_id + str(t)) % 60)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def _guid(seed: str) -> str:
    return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))

def _fallback(ev: dict) -> dict:
    p   = ev.get("provider", "AWS")
    ts  = _ts(ev.get("t", 0), ev.get("scenario_id", ""))
    uid = ev.get("entity_id", "unknown")
    src = ev.get("source_ip", "0.0.0.0")
    act = ev.get("action",    "UnknownAction")
    tgt = ev.get("target_id", "")
    acct= ev.get("cloud_account", "123456789012")
    rgn = ev.get("region",    "us-east-1")
    st  = ev.get("status",    "Success")

    if p == "AWS":
        svc_map = {"AssumeRole": "sts", "RunInstances": "ec2",
                   "GetObject": "s3", "InvokeFunction": "lambda"}
        svc = svc_map.get(act, "iam")
        log = {
            "eventVersion": "1.08", "eventTime": ts,
            "eventSource": f"{svc}.amazonaws.com", "eventName": act,
            "awsRegion": rgn, "sourceIPAddress": src,
            "userAgent": "aws-cli/2.15.0",
            "userIdentity": {"type": "IAMUser", "userName": uid,
                             "arn": f"arn:aws:iam::{acct}:user/{uid}", "accountId": acct},
            "requestParameters": {"resourceId": tgt} if tgt else None,
            "responseElements": None,
            "errorCode": None if st == "Success" else "AccessDenied",
            "requestID": _guid(f"{uid}{ts}req"), "eventID": _guid(f"{uid}{ts}evt"),
            "readOnly": act.startswith("Describe") or act.startswith("Get"),
            "eventType": "AwsApiCall", "recipientAccountId": acct,
        }
    elif p == "Azure":
        op = act if "/" in act else f"Microsoft.Compute/virtualMachines/{act}"
        log = {
            "time": ts,
            "resourceId": f"/subscriptions/{acct}/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/{tgt}",
            "operationName": op, "operationVersion": "2023-07-01",
            "category": "Administrative",
            "resultType": "Succeeded" if st == "Success" else "Failed",
            "resultSignature": "200" if st == "Success" else "403",
            "durationMs": 312, "callerIpAddress": src,
            "correlationId": _guid(f"{uid}{ts}corr"),
            "identity": {"claims": {"name": uid, "upn": f"{uid}@corp.onmicrosoft.com"}},
            "level": "Information", "location": rgn,
            "properties": {"statusCode": "OK" if st == "Success" else "Forbidden",
                           "serviceRequestId": _guid(f"{uid}{ts}svc")},
        }
    else:  # GCP
        method = act if "." in act else f"compute.instances.{act.lower()}"
        log = {
            "logName": f"projects/{acct}/logs/cloudaudit.googleapis.com%2Factivity",
            "resource": {"type": "gce_instance",
                         "labels": {"project_id": acct, "zone": f"{rgn}-a"}},
            "timestamp": ts, "severity": "NOTICE",
            "protoPayload": {
                "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
                "status": {} if st == "Success" else {"code": 7, "message": "PERMISSION_DENIED"},
                "authenticationInfo": {"principalEmail": f"{uid}@{acct}.iam.gserviceaccount.com"},
                "requestMetadata": {"callerIp": src},
                "serviceName": method.split(".")[0] + ".googleapis.com",
                "methodName": method,
                "resourceName": f"projects/{acct}/zones/{rgn}-a/instances/{tgt}",
            },
            "insertId": _sid(f"{uid}{ts}"),
            "receiveTimestamp": ts,
        }

    log["_pipeline_meta"] = _make_meta(ev)
    return log


def _make_meta(ev: dict) -> dict:
    return {
        "edge_id":      ev.get("edge_id", ""),
        "scenario_id":  ev.get("scenario_id", ""),
        "t":            int(ev.get("t", 0)),
        "malicious":    int(ev.get("malicious", 0)),
        "attack_phase": ev.get("attack_phase", "benign"),
        "provider":     ev.get("provider", "AWS"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROQ GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None

def get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=key)
    return _groq_client


def generate_one(ev: dict, max_retries: int = 3) -> dict:
    """
    Call Groq to generate a provider-native log for one structured event.
    Returns {"log": dict, "fallback": bool}.
    """
    client   = get_client()
    provider = ev.get("provider", "AWS")
    system   = SYSTEM_PROMPTS.get(provider, SYSTEM_PROMPTS["AWS"])
    payload  = {k: ev.get(k, "") for k in INPUT_FIELDS if k in ev}
    user_msg = json.dumps(payload, default=str)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user_msg},
                ],
                temperature=0.1,     # near-deterministic — schema task not creative
                max_tokens=800,
                stop=None,
            )
            raw    = resp.choices[0].message.content.strip()
            parsed = extract_json(raw)

            if parsed is None:
                raise ValueError(f"JSON extraction failed. Raw[:150]: {raw[:150]}")

            # Always overwrite _pipeline_meta — never trust the model for labels
            parsed["_pipeline_meta"] = _make_meta(ev)
            return {"log": parsed, "fallback": False}

        except Exception as exc:
            last_err = exc
            # Groq rate limit → back off
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                wait = 10 * (attempt + 1)
                print(f"  Rate limit hit, waiting {wait}s …")
                time.sleep(wait)
            else:
                time.sleep(1)

    # All retries failed → deterministic fallback
    print(f"  ⚠️  Groq failed after {max_retries} attempts: {last_err}. Using fallback.")
    return {"log": _fallback(ev), "fallback": True}


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP  (for Render deployment)
# ─────────────────────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _validate(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected = API_KEY or os.environ.get("STAGE_0B_API_KEY", "")
    if not expected:
        return api_key or ""   # no key configured → open (dev mode)
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(expected.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


class EventRequest(BaseModel):
    provider:      str = Field("AWS")
    action:        str = Field(...)
    entity_id:     str = Field(...)
    entity_type:   str = Field("User")
    target_id:     str = Field("")
    source_ip:     str = Field("0.0.0.0")
    region:        str = Field("us-east-1")
    cloud_account: str = Field("")
    status:        str = Field("Success")
    malicious:     int = Field(0)
    attack_phase:  str = Field("benign")
    edge_id:       str = Field("")
    scenario_id:   str = Field("")
    t:             int = Field(0)

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, v):
        if v not in ("AWS", "Azure", "GCP"):
            raise ValueError("provider must be AWS, Azure, or GCP")
        return v


class BatchRequest(BaseModel):
    events: list[EventRequest]


web = FastAPI(
    title="Trinetra — Stage 0b SIEM Log Generator (Groq)",
    version="4.0.0",
    description="Converts Stage 0a structured events into provider-native cloud audit logs via Groq LLaMA-3.1-8B.",
)
web.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@web.get("/health")
def health():
    return {"stage": "0b", "backend": "groq",
            "model": GROQ_MODEL, "status": "ok", "version": "4.0.0"}


@web.post("/generate")
def generate(req: EventRequest, _: str = Depends(_validate)):
    return generate_one(req.model_dump())


@web.post("/generate_batch")
def generate_batch(req: BatchRequest, _: str = Depends(_validate)):
    results, fallbacks = [], 0
    for ev in req.events:
        r = generate_one(ev.model_dump())
        results.append(r)
        if r["fallback"]:
            fallbacks += 1
        time.sleep(REQUEST_DELAY)   # respect rate limit between calls
    n = len(results)
    return {
        "total": n, "successful": n,
        "fallback_count": fallbacks,
        "fallback_rate":  round(fallbacks / n, 3) if n else 0,
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH GENERATION SCRIPT  (run directly: python app.py --batch …)
# Reads structured_events.parquet from Stage 0a, writes 4 parquets for Stage 1
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(input_path: str, output_dir: str, checkpoint_every: int = 500):
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined_path   = os.path.join(output_dir, "stage0b_combined.parquet")
    checkpoint_path = os.path.join(output_dir, "_checkpoint.parquet")

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} events from {input_path}")
    print(f"Provider counts:\n{df['provider'].value_counts().to_string()}")

    # Resume from checkpoint if it exists
    already_done = set()
    done_rows    = []
    if os.path.exists(checkpoint_path):
        ckpt_df = pd.read_parquet(checkpoint_path)
        already_done = set(zip(ckpt_df["scenario_id"], ckpt_df["t"], ckpt_df["edge_id"]))
        done_rows    = ckpt_df.to_dict("records")
        print(f"Resuming: {len(done_rows):,} already processed, {len(df)-len(done_rows):,} remaining")

    events     = df.to_dict("records")
    results    = list(done_rows)
    fallbacks  = 0

    for ev in tqdm(events):
        key = (str(ev.get("scenario_id","")), str(ev.get("t","")), str(ev.get("edge_id","")))
        if key in already_done:
            continue

        r = generate_one(ev)
        if r["fallback"]:
            fallbacks += 1

        meta = r["log"].get("_pipeline_meta", _make_meta(ev))
        results.append({
            "raw_log":      json.dumps(r["log"], separators=(",",":"), default=str),
            "edge_id":      meta.get("edge_id", ""),
            "scenario_id":  meta.get("scenario_id", ""),
            "t":            meta.get("t", 0),
            "malicious":    meta.get("malicious", 0),
            "attack_phase": meta.get("attack_phase", "benign"),
            "provider":     meta.get("provider", "AWS"),
            "_fallback":    r["fallback"],
        })
        already_done.add(key)
        time.sleep(REQUEST_DELAY)

        # Save checkpoint every N rows
        if len(results) % checkpoint_every == 0:
            pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
            print(f"  Checkpoint saved ({len(results):,} done, {fallbacks} fallbacks)")

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_parquet(combined_path, index=False)

    for prov in ("AWS", "Azure", "GCP"):
        sub = out_df[out_df["provider"] == prov]
        sub.to_parquet(os.path.join(output_dir, f"stage0b_{prov.lower()}.parquet"), index=False)
        print(f"  {prov}: {len(sub):,} rows")

    print(f"\nDone. {len(out_df):,} total logs, {fallbacks} fallbacks ({100*fallbacks/len(out_df):.1f}%)")
    print(f"Outputs in: {output_dir}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-GEN  — calls THIS app's /generate_batch endpoint to produce
#             300 logs per provider → aws_logs.parquet, azure_logs.parquet,
#             gcp_logs.parquet, all_logs.parquet   (Stage 1 input)
#
# Usage (from any machine — Render must be running):
#   python app.py --selfgen
#   python app.py --selfgen --url https://stage0b-attack-sim.onrender.com
#                           --apikey cb84a4ee...
#                           --output ./outputs
#                           --batch-size 25
# ─────────────────────────────────────────────────────────────────────────────

# ── 300-event corpus: 100 per provider, varied actions / phases / IPs ────────
def _build_event_corpus() -> list[dict]:
    """
    Returns a list of 300 structured events (100 AWS, 100 Azure, 100 GCP).
    Every event is a valid EventRequest payload.  Values are varied enough
    that Groq produces semantically diverse logs rather than near-duplicates.
    """
    import random, itertools
    rng = random.Random(42)   # reproducible

    # ── shared pools ──────────────────────────────────────────────────────────
    users       = [f"user_{n}" for n in ["alice","bob","carol","dave","eve",
                                          "frank","grace","henry","ivan","julia"]]
    mal_ips     = ["185.220.101.45","45.142.212.100","198.51.100.23",
                   "103.21.244.0","91.108.4.0","192.0.2.77","203.0.113.99"]
    benign_ips  = ["10.0.0.5","10.0.1.12","172.16.0.8","192.168.1.100",
                   "10.10.10.50","172.31.5.3","10.0.2.20"]
    scenarios   = [f"sc_{i:03d}" for i in range(1, 51)]   # sc_001 … sc_050
    attack_phases = ["privilege_escalation","lateral_movement",
                     "cross_cloud_pivot","cve_exploitation"]

    def _meta(i: int, provider: str, malicious: int, phase: str) -> dict:
        sid = rng.choice(scenarios)
        ip  = rng.choice(mal_ips if malicious else benign_ips)
        return {
            "edge_id":      f"e_{i:04d}",
            "scenario_id":  sid,
            "t":            rng.randint(0, 19),
            "malicious":    malicious,
            "attack_phase": phase,
            "source_ip":    ip,
            "entity_id":    rng.choice(users),
            "entity_type":  "User",
            "status":       "Success" if malicious == 0 or rng.random() > 0.3 else "Failure",
            "provider":     provider,
        }

    # ── AWS events (300) ──────────────────────────────────────────────────────
    aws_templates = [
        # (action, target_id, region, cloud_account, malicious_weight)
        ("AssumeRole",          "admin_role",           "us-east-1",  "123456789012", 0.6),
        ("AssumeRole",          "devops_role",          "us-west-2",  "123456789012", 0.4),
        ("GetObject",           "prod-data-bucket",     "us-west-2",  "123456789012", 0.1),
        ("GetObject",           "backup-bucket",        "eu-west-1",  "234567890123", 0.1),
        ("PutObject",           "log-archive-bucket",   "us-east-1",  "123456789012", 0.2),
        ("RunInstances",        "vm_001",               "eu-west-1",  "123456789012", 0.5),
        ("RunInstances",        "vm_009",               "ap-south-1", "234567890123", 0.5),
        ("DescribeInstances",   "vpc_001",              "us-east-1",  "123456789012", 0.0),
        ("CreateUser",          "svc-account-new",      "us-east-1",  "123456789012", 0.7),
        ("AttachUserPolicy",    "arn:aws:iam::aws:policy/AdministratorAccess",
                                                        "us-east-1",  "123456789012", 0.9),
        ("DeleteTrail",         "mgmt-trail",           "us-east-1",  "123456789012", 0.95),
        ("StopLogging",         "mgmt-trail",           "us-east-1",  "123456789012", 0.95),
        ("InvokeFunction",      "data-exfil-fn",        "us-east-1",  "123456789012", 0.8),
        ("InvokeFunction",      "notify-fn",            "us-west-2",  "123456789012", 0.1),
        ("ListBuckets",         "",                     "us-east-1",  "123456789012", 0.3),
        ("GetSecretValue",      "prod/db-password",     "us-east-1",  "123456789012", 0.6),
        ("DescribeSecurityGroups","sg-001",             "us-east-1",  "123456789012", 0.2),
        ("AuthorizeSecurityGroupIngress","sg-001",      "us-east-1",  "123456789012", 0.7),
        ("CreateAccessKey",     "user_alice",           "us-east-1",  "123456789012", 0.8),
        ("ConsoleLogin",        "",                     "us-east-1",  "123456789012", 0.3),
    ]
    aws_events = []
    for i, tmpl in enumerate(itertools.islice(itertools.cycle(aws_templates), 300)):
        action, target, region, acct, mw = tmpl
        mal = 1 if rng.random() < mw else 0
        phase = rng.choice(attack_phases) if mal else "benign"
        ev = _meta(i + 1, "AWS", mal, phase)
        ev.update({"action": action, "target_id": target,
                   "region": region, "cloud_account": acct})
        aws_events.append(ev)

    # ── Azure events (300) ────────────────────────────────────────────────────
    azure_templates = [
        ("Microsoft.Compute/virtualMachines/start",   "vm_005", "eastus",       "sub-prod-001", 0.5),
        ("Microsoft.Compute/virtualMachines/delete",  "vm_012", "westeurope",   "sub-prod-001", 0.7),
        ("Microsoft.Compute/virtualMachines/write",   "vm_007", "eastus2",      "sub-dev-002",  0.4),
        ("Microsoft.Authorization/roleAssignments/write","admin-role","eastus",  "sub-prod-001", 0.85),
        ("Microsoft.Authorization/roleAssignments/delete","reader-role","eastus","sub-prod-001", 0.6),
        ("Microsoft.KeyVault/vaults/secrets/read",    "kv-prod","eastus",        "sub-prod-001", 0.5),
        ("Microsoft.KeyVault/vaults/secrets/write",   "kv-prod","eastus",        "sub-prod-001", 0.7),
        ("Microsoft.Storage/storageAccounts/read",    "sa001",  "eastus",        "sub-prod-001", 0.1),
        ("Microsoft.Storage/storageAccounts/write",   "sa002",  "westus",        "sub-prod-001", 0.3),
        ("Microsoft.Network/networkSecurityGroups/write","nsg-001","eastus",     "sub-prod-001", 0.6),
        ("Microsoft.Network/virtualNetworks/read",    "vnet-01","eastus",        "sub-prod-001", 0.1),
        ("Microsoft.Sql/servers/databases/read",      "db-prod","eastus",        "sub-prod-001", 0.2),
        ("Microsoft.Sql/servers/firewallRules/write", "db-prod","eastus",        "sub-prod-001", 0.75),
        ("Microsoft.Web/sites/restart",               "webapp-01","eastus",      "sub-dev-002",  0.3),
        ("Microsoft.ContainerRegistry/registries/push","acr-01","eastus",        "sub-prod-001", 0.4),
        ("Microsoft.Insights/diagnosticSettings/delete","audit-diag","eastus",   "sub-prod-001", 0.9),
        ("Microsoft.AAD/users/write",                 "user_carol","eastus",     "sub-prod-001", 0.5),
        ("Microsoft.AAD/servicePrincipals/write",     "svc-exfil","eastus",      "sub-prod-001", 0.8),
        ("Microsoft.Compute/snapshots/write",         "snap-001","eastus",       "sub-prod-001", 0.5),
        ("Microsoft.Resources/deployments/write",     "deploy-01","eastus",      "sub-prod-001", 0.3),
    ]
    azure_entities = ["john.doe@corp.com","alice.smith@corp.com","bob.jones@corp.com",
                      "carol.white@corp.com","dave.brown@corp.com","svc-principal@corp.com"]
    azure_events = []
    for i, tmpl in enumerate(itertools.islice(itertools.cycle(azure_templates), 300)):
        action, target, region, acct, mw = tmpl
        mal = 1 if rng.random() < mw else 0
        phase = rng.choice(attack_phases) if mal else "benign"
        ev = _meta(i + 301, "Azure", mal, phase)
        ev.update({"action": action, "target_id": target,
                   "region": region, "cloud_account": acct,
                   "entity_id": rng.choice(azure_entities)})
        azure_events.append(ev)

    # ── GCP events (300) ──────────────────────────────────────────────────────
    gcp_templates = [
        ("compute.instances.insert",        "vm_001", "us-central1",    "gcp-prod-project",  0.4),
        ("compute.instances.delete",        "vm_003", "us-east1",       "gcp-prod-project",  0.6),
        ("compute.instances.setMetadata",   "vm_008", "us-central1",    "gcp-prod-project",  0.8),
        ("compute.instances.start",         "vm_002", "europe-west1",   "gcp-dev-project",   0.2),
        ("compute.instances.stop",          "vm_004", "us-central1",    "gcp-prod-project",  0.3),
        ("iam.serviceAccounts.create",      "svc-exfil","us-central1",  "gcp-prod-project",  0.85),
        ("iam.serviceAccounts.setIamPolicy","svc-admin","us-central1",  "gcp-prod-project",  0.9),
        ("iam.serviceAccountKeys.create",   "svc-001","us-central1",    "gcp-prod-project",  0.8),
        ("storage.buckets.list",            "",       "us-central1",    "gcp-prod-project",  0.2),
        ("storage.objects.get",             "bucket-prod","us",         "gcp-prod-project",  0.1),
        ("storage.objects.create",          "bucket-exfil","us",        "gcp-prod-project",  0.7),
        ("cloudfunctions.functions.call",   "fn-exec","us-central1",   "gcp-dev-project",   0.5),
        ("cloudsql.instances.update",       "db-prod","us-central1",   "gcp-prod-project",  0.4),
        ("container.clusters.create",       "k8s-001","us-central1",   "gcp-prod-project",  0.3),
        ("container.clusters.delete",       "k8s-002","us-central1",   "gcp-prod-project",  0.6),
        ("logging.sinks.delete",            "audit-sink","global",      "gcp-prod-project",  0.9),
        ("cloudkms.cryptoKeyVersions.destroy","key-001","us-central1", "gcp-prod-project",  0.8),
        ("bigquery.tables.create",          "dataset-01","us",         "gcp-prod-project",  0.2),
        ("bigquery.tables.getData",         "dataset-01","us",         "gcp-prod-project",  0.3),
        ("resourcemanager.projects.setIamPolicy","gcp-prod-project","global","gcp-prod-project",0.9),
    ]
    gcp_entities = [f"{u}@gcp-prod-project.iam.gserviceaccount.com" for u in
                    ["svc-worker","svc-admin","mal-ip-node","svc-exfil","svc-deploy"]] + \
                   [f"{u}@corp.com" for u in ["alice","bob","carol","dave","eve"]]
    gcp_events = []
    for i, tmpl in enumerate(itertools.islice(itertools.cycle(gcp_templates), 300)):
        action, target, region, acct, mw = tmpl
        mal = 1 if rng.random() < mw else 0
        phase = rng.choice(attack_phases) if mal else "benign"
        ev = _meta(i + 601, "GCP", mal, phase)
        ev.update({"action": action, "target_id": target,
                   "region": region, "cloud_account": acct,
                   "entity_id": rng.choice(gcp_entities)})
        gcp_events.append(ev)

    # Shuffle within each provider so attack/benign events interleave
    rng.shuffle(aws_events)
    rng.shuffle(azure_events)
    rng.shuffle(gcp_events)

    return aws_events + azure_events + gcp_events


def run_selfgen(
    base_url:   str = "https://stage0b-attack-sim.onrender.com",
    api_key:    str = "",
    output_dir: str = "./stage0b_outputs",
    batch_size: int = 25,
):
    """
    Calls /generate_batch on the running Render service to produce
    300 logs per provider, then saves:
        aws_logs.parquet  /  azure_logs.parquet  /  gcp_logs.parquet  /  all_logs.parquet
    """
    import requests, pandas as pd
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url     = base_url.rstrip("/") + "/generate_batch"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    events  = _build_event_corpus()          # 300 structured events
    total   = len(events)
    rows    = []
    fallbacks = 0

    print(f"Stage 0b self-gen  →  {url}")
    print(f"Events: {total} ({total//3} per provider)  |  batch_size={batch_size}")
    print("-" * 60)

    for start in range(0, total, batch_size):
        chunk = events[start : start + batch_size]
        print(f"  Sending batch {start//batch_size + 1} "
              f"({start+1}–{min(start+batch_size, total)}/{total}) …", end=" ", flush=True)

        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"events": chunk},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"ERROR — {exc}")
            # On HTTP failure, build fallback rows for the whole chunk
            for ev in chunk:
                fb_log = _fallback(ev)
                meta   = _make_meta(ev)
                rows.append({
                    "raw_log":      json.dumps(fb_log, separators=(",",":"), default=str),
                    "provider":     meta["provider"],
                    "scenario_id":  meta["scenario_id"],
                    "t":            meta["t"],
                    "malicious":    meta["malicious"],
                    "attack_phase": meta["attack_phase"],
                    "_fallback":    True,
                })
                fallbacks += 1
            continue

        fb_in_batch = data.get("fallback_count", 0)
        fallbacks  += fb_in_batch
        print(f"ok  (fallbacks={fb_in_batch})")

        for result, ev in zip(data["results"], chunk):
            log  = result["log"]
            meta = log.get("_pipeline_meta", _make_meta(ev))
            rows.append({
                "raw_log":      json.dumps(log, separators=(",",":"), default=str),
                "provider":     meta.get("provider",     ev["provider"]),
                "scenario_id":  meta.get("scenario_id",  ev["scenario_id"]),
                "t":            meta.get("t",             ev["t"]),
                "malicious":    meta.get("malicious",     ev["malicious"]),
                "attack_phase": meta.get("attack_phase",  ev["attack_phase"]),
                "_fallback":    result.get("fallback", False),
            })

    # ── Save parquets ──────────────────────────────────────────────────────────
    all_df = pd.DataFrame(rows)

    provider_map = {"AWS": "aws_logs.parquet",
                    "Azure": "azure_logs.parquet",
                    "GCP": "gcp_logs.parquet"}

    for prov, fname in provider_map.items():
        sub = all_df[all_df["provider"] == prov].reset_index(drop=True)
        out = os.path.join(output_dir, fname)
        sub.to_parquet(out, index=False)
        size_kb = os.path.getsize(out) / 1024
        print(f"  ✓  {fname:<22}  {len(sub):>3} rows  {size_kb:.1f} KB")

    all_path = os.path.join(output_dir, "all_logs.parquet")
    all_df.to_parquet(all_path, index=False)
    size_kb = os.path.getsize(all_path) / 1024
    print(f"  ✓  {'all_logs.parquet':<22}  {len(all_df):>3} rows  {size_kb:.1f} KB")

    fb_pct = 100 * fallbacks / len(all_df) if all_df is not None and len(all_df) else 0
    print(f"\nDone — {len(all_df)} logs total  |  fallbacks: {fallbacks} ({fb_pct:.1f}%)")
    print(f"Output dir: {os.path.abspath(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trinetra Stage 0b")
    sub = parser.add_subparsers(dest="cmd")

    # ── selfgen ───────────────────────────────────────────────────────────────
    sg = sub.add_parser("selfgen",
        help="Call the running Render service to generate 300 logs/provider → 4 parquets")
    sg.add_argument("--url",        default="https://stage0b-attack-sim.onrender.com",
                    help="Base URL of the Render deployment")
    sg.add_argument("--apikey",     default=os.environ.get("STAGE_0B_API_KEY", ""),
                    help="X-API-Key header value (or set STAGE_0B_API_KEY env var)")
    sg.add_argument("--output",     default="./stage0b_outputs",
                    help="Directory to write parquet files into")
    sg.add_argument("--batch-size", type=int, default=25,
                    help="Events per /generate_batch call (default 25)")

    # ── batch (legacy — needs structured_events.parquet from Stage 0a) ────────
    bt = sub.add_parser("batch",
        help="(Legacy) Read structured_events.parquet and generate via Groq directly")
    bt.add_argument("--input",  default="structured_events.parquet")
    bt.add_argument("--output", default="./stage0b_outputs")
    bt.add_argument("--checkpoint-every", type=int, default=500)

    # ── serve ─────────────────────────────────────────────────────────────────
    sv = sub.add_parser("serve", help="Run the FastAPI server locally")
    sv.add_argument("--port", type=int, default=8000)

    # ── backward-compat flat flags (old: python app.py --batch / --serve) ─────
    parser.add_argument("--batch",  action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input",  default="structured_events.parquet",
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", default="./stage0b_outputs",
                        help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help=argparse.SUPPRESS)
    parser.add_argument("--serve",  action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port",   type=int, default=8000, help=argparse.SUPPRESS)
    parser.add_argument("--selfgen",action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--url",    default="https://stage0b-attack-sim.onrender.com",
                        help=argparse.SUPPRESS)
    parser.add_argument("--apikey", default=os.environ.get("STAGE_0B_API_KEY", ""),
                        help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=25, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # subcommand routing
    if args.cmd == "selfgen":
        run_selfgen(args.url, args.apikey, args.output, args.batch_size)
    elif args.cmd == "batch":
        run_batch(args.input, args.output, args.checkpoint_every)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("app:web", host="0.0.0.0", port=args.port, reload=False)
    # flat-flag backward compat
    elif getattr(args, "selfgen", False):
        run_selfgen(args.url, args.apikey, args.output, args.batch_size)
    elif getattr(args, "batch", False):
        run_batch(args.input, args.output, args.checkpoint_every)
    elif getattr(args, "serve", False):
        import uvicorn
        uvicorn.run("app:web", host="0.0.0.0", port=args.port, reload=False)
    else:
        parser.print_help()