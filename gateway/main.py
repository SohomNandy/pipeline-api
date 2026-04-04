"""
Pipeline API Gateway — All Stages 0b through 10
Single entry point for the entire Trinetra threat detection pipeline.
"""
import os, sys
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
sys.path.append("..")
from shared.auth import get_api_key_validator

app = FastAPI(title="Trinetra Pipeline Gateway", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# All stage URLs — set as Render environment variables
STAGE_URLS = {
    "0B": os.environ.get("STAGE_0B_URL",  ""),   # Modal GPU
    "1":  os.environ.get("STAGE_1_URL",   ""),   # Render CPU
    "2":  os.environ.get("STAGE_2_URL",   ""),   # Modal CPU
    "3A": os.environ.get("STAGE_3A_URL",  ""),   # Render CPU
    "3B": os.environ.get("STAGE_3B_URL",  ""),   # Modal CPU
    "4":  os.environ.get("STAGE_4_URL",   ""),   # Render CPU
    "5":  os.environ.get("STAGE_5_URL",   ""),   # Render CPU
    "67": os.environ.get("STAGE_67_URL",  ""),   # Modal GPU (parallel 6+7)
    "8":  os.environ.get("STAGE_8_URL",   ""),   # Render/Modal
    "9":  os.environ.get("STAGE_9_URL",   ""),   # Render CPU
    "10": os.environ.get("STAGE_10_URL",  ""),   # Modal GPU
}

validate_master = get_api_key_validator("GATEWAY")


async def proxy(stage: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    url = STAGE_URLS.get(stage, "")
    if not url:
        raise HTTPException(status_code=503, detail=f"Stage {stage} not configured")
    stage_key = os.environ.get(f"STAGE_{stage}_API_KEY", "")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{url.rstrip('/')}/{path}",
            json=payload,
            headers={"X-API-Key": stage_key, "Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "pipeline": "0b → 1 → 2 → 3a → 3b → 4 → 5 → 6+7 → 8 → 9 → 10",
        "stages":   {k: "configured" if v else "not configured" for k, v in STAGE_URLS.items()},
    }


# ── Stage routes ──────────────────────────────────────────────────────────────
@app.post("/stage0b/generate")
async def stage0b(request: Request, _=Depends(validate_master)):
    return await proxy("0B", "generate", await request.json(), timeout=180.0)


@app.post("/stage1/normalise")
async def stage1(request: Request, _=Depends(validate_master)):
    return await proxy("1", "normalise", await request.json())


@app.post("/stage2/embed")
async def stage2(request: Request, _=Depends(validate_master)):
    return await proxy("2", "embed", await request.json(), timeout=60.0)


@app.post("/stage2/embed_batch")
async def stage2_batch(request: Request, _=Depends(validate_master)):
    return await proxy("2", "embed_batch", await request.json(), timeout=120.0)


@app.post("/stage3a/extract")
async def stage3a(request: Request, _=Depends(validate_master)):
    return await proxy("3A", "extract", await request.json())


@app.post("/stage3b/score")
async def stage3b(request: Request, _=Depends(validate_master)):
    return await proxy("3B", "score", await request.json(), timeout=60.0)


@app.post("/stage4/embed")
async def stage4(request: Request, _=Depends(validate_master)):
    return await proxy("4", "embed", await request.json())


@app.post("/stage4/embed_batch")
async def stage4_batch(request: Request, _=Depends(validate_master)):
    return await proxy("4", "embed_batch", await request.json())


@app.post("/stage5/build_graph")
async def stage5(request: Request, _=Depends(validate_master)):
    return await proxy("5", "build_graph", await request.json(), timeout=60.0)


@app.post("/stage67/predict")
async def stage67(request: Request, _=Depends(validate_master)):
    # Parallel Stage 6 + 7 — longer timeout for GRU over 20 snapshots
    return await proxy("67", "predict", await request.json(), timeout=300.0)


@app.post("/stage8/predict")
async def stage8(request: Request, _=Depends(validate_master)):
    return await proxy("8", "predict/fusion", await request.json())


@app.post("/stage9/explain")
async def stage9(request: Request, _=Depends(validate_master)):
    return await proxy("9", "explain", await request.json())


@app.post("/stage10/generate")
async def stage10(request: Request, _=Depends(validate_master)):
    # Mistral-7B generation — long timeout
    return await proxy("10", "generate", await request.json(), timeout=300.0)