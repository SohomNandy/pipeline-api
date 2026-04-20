import modal, os, json, torch, asyncio, time, hashlib, secrets
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from collections import Counter
import random
import numpy as np

# ============================================================
# MODAL IMAGE (no extra ML deps – sampling uses built-ins)
# ============================================================
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

# ============================================================
# LIGHTWEIGHT SAMPLER (no pandas/sklearn/sentence-transformers)
# ============================================================
class LightweightSampler:
    """Smart sampling using only built-in Python – runs inside Modal"""
    
    @staticmethod
    def temporal_sample(events: List[dict], timestep_key='t', scenario_key='scenario_id'):
        """Keep early, peak, and late timesteps per scenario"""
        by_scenario = {}
        for e in events:
            sid = e.get(scenario_key, 'default')
            by_scenario.setdefault(sid, []).append(e)
        
        sampled = []
        for sid, evts in by_scenario.items():
            # Group by timestep
            by_t = {}
            for e in evts:
                t = e.get(timestep_key, 0)
                by_t.setdefault(t, []).append(e)
            
            timesteps = sorted(by_t.keys())
            if len(timesteps) <= 10:
                sampled.extend(evts)
                continue
            
            # Early (first 3 timesteps)
            for t in timesteps[:3]:
                sampled.extend(by_t[t])
            
            # Peak (timestep with most malicious)
            malicious_counts = {}
            for t, evt_list in by_t.items():
                mal_count = sum(1 for e in evt_list if e.get('malicious', 0) == 1)
                malicious_counts[t] = mal_count
            if malicious_counts:
                peak_t = max(malicious_counts, key=malicious_counts.get)
                sampled.extend(by_t[peak_t])
            
            # Late (last 3 timesteps)
            for t in timesteps[-3:]:
                sampled.extend(by_t[t])
        
        return sampled
    
    @staticmethod
    def phase_balanced_sample(events: List[dict], phase_key='attack_phase', target_per_phase=30):
        """Ensure rare phases are oversampled (simple duplication)"""
        by_phase = {}
        for e in events:
            phase = e.get(phase_key, 'benign')
            by_phase.setdefault(phase, []).append(e)
        
        sampled = []
        for phase, evts in by_phase.items():
            if len(evts) < target_per_phase and len(evts) > 0:
                # Oversample with duplication
                n_copies = (target_per_phase // len(evts)) + 1
                for _ in range(n_copies):
                    sampled.extend(evts)
                sampled = sampled[:target_per_phase]
            else:
                sampled.extend(evts)
        
        return sampled
    
    @staticmethod
    def provider_balanced_sample(events: List[dict], provider_key='provider', min_per_provider=30):
        """Ensure each cloud provider is represented"""
        by_provider = {}
        for e in events:
            prov = e.get(provider_key, 'AWS')
            by_provider.setdefault(prov, []).append(e)
        
        sampled = []
        for prov, evts in by_provider.items():
            if len(evts) < min_per_provider and len(evts) > 0:
                # Oversample with replacement
                n_needed = min_per_provider - len(evts)
                sampled.extend(evts)
                for _ in range(n_needed):
                    sampled.append(random.choice(evts))
            else:
                sampled.extend(evts)
        
        return sampled
    
    @staticmethod
    def sample(events: List[dict], target_size: int = 2000) -> List[dict]:
        """Run all sampling strategies in sequence"""
        if len(events) <= target_size:
            return events
        
        # Step 1: Temporal sampling
        sampled = LightweightSampler.temporal_sample(events)
        
        # Step 2: Phase balancing
        sampled = LightweightSampler.phase_balanced_sample(sampled)
        
        # Step 3: Provider balancing
        sampled = LightweightSampler.provider_balanced_sample(sampled)
        
        # Step 4: Final random sample to target size
        if len(sampled) > target_size:
            sampled = random.sample(sampled, target_size)
        
        return sampled


# ============================================================
# MODAL CLASS (SIEM Generator)
# ============================================================
@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
class SIEMGenerator:
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login as hf_login

        hf_login(token=os.environ["HF_TOKEN"])

        BASE_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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
            {"role": "user", "content": json.dumps(event)},
        ]
        text = self.tokenizer.apply_chat_template(
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

    @modal.method()
    def generate_batch(self, events: List[dict]) -> List[dict]:
        results = []
        for event in events:
            results.append(self.generate(event))
        return results
    
    @modal.method()
    def generate_with_sampling(self, events: List[dict], target_size: int = 2000) -> List[dict]:
        """Smart sampling + generation in one call"""
        sampled_events = LightweightSampler.sample(events, target_size)
        return self.generate_batch(sampled_events)


# ============================================================
# FASTAPI APP
# ============================================================
@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    web = FastAPI(title="Stage 0b - SIEM Log Generator")
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

    # ----- Request/Response Models -----
    class GenerateRequest(BaseModel):
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

    class BatchGenerateRequest(BaseModel):
        events: List[GenerateRequest] = Field(..., max_items=200)
        apply_sampling: bool = Field(False, description="Apply smart sampling before generation")
        target_size: int = Field(2000, ge=100, le=5000)

    class BatchStatusResponse(BaseModel):
        batch_id: str
        status: str
        total: int
        completed: int
        failed: int
        results: Optional[List[dict]] = None

    # Simple in-memory store for async batches
    batch_store: Dict[str, dict] = {}

    # ----- Endpoints -----
    @web.get("/health")
    async def health():
        return {
            "stage": "0b",
            "status": "ok",
            "model": "sohomn/siem-log-generator-llama31-8b",
            "batch_supported": True,
            "sampling_supported": True,
            "max_batch_size": 200
        }

    @web.get("/")
    async def root():
        return {
            "service": "Stage 0b - SIEM Log Generator",
            "endpoints": {
                "GET /health": "Health check",
                "POST /generate": "Generate single log",
                "POST /generate_batch": "Generate batch (sync)",
                "POST /generate_batch_async": "Generate batch (async)",
                "POST /generate_with_sampling": "Smart sampling + generation"
            }
        }

    @web.post("/generate")
    async def generate(req: GenerateRequest, _=Depends(validate)):
        result = generator.generate.remote(req.dict())
        if "error" in result:
            raise HTTPException(status_code=500, detail=result)
        return result

    @web.post("/generate_batch")
    async def generate_batch(req: BatchGenerateRequest, _=Depends(validate)):
        events_dict = [e.dict() for e in req.events]
        
        if req.apply_sampling:
            results = generator.generate_with_sampling.remote(events_dict, req.target_size)
        else:
            results = generator.generate_batch.remote(events_dict)
        
        return {
            "total": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results
        }

    @web.post("/generate_batch_async")
    async def generate_batch_async(
        req: BatchGenerateRequest,
        background_tasks: BackgroundTasks,
        _=Depends(validate)
    ):
        batch_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        
        batch_store[batch_id] = {
            "status": "processing",
            "total": len(req.events),
            "completed": 0,
            "failed": 0,
            "results": None
        }
        
        async def process():
            events_dict = [e.dict() for e in req.events]
            if req.apply_sampling:
                results = generator.generate_with_sampling.remote(events_dict, req.target_size)
            else:
                results = generator.generate_batch.remote(events_dict)
            
            batch_store[batch_id] = {
                "status": "completed",
                "total": len(results),
                "completed": sum(1 for r in results if "error" not in r),
                "failed": sum(1 for r in results if "error" in r),
                "results": results
            }
        
        background_tasks.add_task(process)
        
        return {
            "batch_id": batch_id,
            "status": "processing",
            "message": "Batch processing started. GET /batch_status/{batch_id}",
            "total": len(req.events)
        }

    @web.post("/generate_with_sampling")
    async def generate_with_sampling(req: BatchGenerateRequest, _=Depends(validate)):
        """One-shot: smart sampling + generation, returns results"""
        events_dict = [e.dict() for e in req.events]
        results = generator.generate_with_sampling.remote(events_dict, req.target_size)
        
        return {
            "original_count": len(req.events),
            "sampled_count": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results
        }

    @web.get("/batch_status/{batch_id}")
    async def batch_status(batch_id: str, _=Depends(validate)):
        if batch_id not in batch_store:
            raise HTTPException(status_code=404, detail="Batch not found")
        return batch_store[batch_id]

    return web