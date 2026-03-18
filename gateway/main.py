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
    "0B": os.environ.get("STAGE_0B_URL",  ""),
    "1":  os.environ.get("STAGE_1_URL",   ""),
    "2":  os.environ.get("STAGE_2_URL",   ""),
    "3A": os.environ.get("STAGE_3A_URL",  ""),
    "3B": os.environ.get("STAGE_3B_URL",  ""),
    "4":  os.environ.get("STAGE_4_URL",   ""),
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
        "pipeline": "0b → 1 → 2 → 3a → 3b → 4 → 5 → 6 → 7 → 8 → 9 → 10",
        "stages": {
            k: "configured" if v else "not configured"
            for k, v in STAGE_URLS.items()
        },
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


@app.post("/stage3a/extract")
async def stage3a_extract(request: Request, _=Depends(validate_master)):
    return await proxy("3A", "extract", await request.json())


@app.post("/stage3b/score")
async def stage3b_score(request: Request, _=Depends(validate_master)):
    return await proxy("3B", "score", await request.json())


@app.post("/stage4/embed")
async def stage4_embed(request: Request, _=Depends(validate_master)):
    return await proxy("4", "embed", await request.json())