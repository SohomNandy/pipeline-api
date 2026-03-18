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
