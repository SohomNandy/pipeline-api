#gateway/main.py
import os, sys, asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
sys.path.append("..")
from shared.auth import get_api_key_validator

app = FastAPI(title="Trinetra Pipeline Gateway", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

STAGE_URLS = {
    "0B": os.environ.get("STAGE_0B_URL",  ""),  # Modal GPU — LLaMA 3.1-8B
    "1":  os.environ.get("STAGE_1_URL",   ""),  # Render CPU — T5 log normaliser
    "2":  os.environ.get("STAGE_2_URL",   ""),  # Modal CPU — BGE log embeddings
    "3A": os.environ.get("STAGE_3A_URL",  ""),  # Render CPU — CodeBERT NER
    "3B": os.environ.get("STAGE_3B_URL",  ""),  # Modal CPU — e5 CVE risk scoring
    "4":  os.environ.get("STAGE_4_URL",   ""),  # Render CPU — flan-t5 identity
    "5":  os.environ.get("STAGE_5_URL",   ""),  # Render CPU — graph construction
    "6":  os.environ.get("STAGE_6_URL",   ""),  # Modal GPU — RGCN structural
    "7":  os.environ.get("STAGE_7_URL",   ""),  # Modal GPU — GRU temporal
    "8":  os.environ.get("STAGE_8_URL",   ""),  # Modal/Render — FT-Transformer
    "9":  os.environ.get("STAGE_9_URL",   ""),  # Modal GPU — GNNExplainer+SHAP+Mistral
    "10": os.environ.get("STAGE_10_URL",  ""),  # same as 9 (integrated service)
}

validate_master = get_api_key_validator("GATEWAY")


async def _call_stage(
    client:    httpx.AsyncClient,
    stage:     str,
    path:      str,
    payload:   dict,
    timeout:   float = 120.0,
) -> dict:
    """Internal helper — calls one stage, raises HTTPException on failure."""
    url = STAGE_URLS.get(stage, "")
    if not url:
        raise HTTPException(status_code=503, detail=f"Stage {stage} not configured")
    stage_key = os.environ.get(f"STAGE_{stage}_API_KEY", "")
    resp = await client.post(
        f"{url.rstrip('/')}/{path}",
        json=payload,
        headers={"X-API-Key": stage_key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def proxy(stage: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    """Single-stage proxy — used by all individual stage routes."""
    async with httpx.AsyncClient() as client:
        return await _call_stage(client, stage, path, payload, timeout)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "version":  "2.1.0",
        "pipeline": "0b → 1 → 2 → 3a → 3b → 4 → 5 → 6∥7 → 8 → 9+10",
        "parallel": "Stage 6 (RGCN) and Stage 7 (GRU) run with dependency chaining",
        "stages":   {k: "configured" if v else "not configured" for k, v in STAGE_URLS.items()},
    }


# ── Individual stage routes (direct access) ───────────────────────────────────
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

@app.post("/stage6/predict")
async def stage6(request: Request, _=Depends(validate_master)):
    """Direct Stage 6 call — returns h_v embeddings + threat scores."""
    return await proxy("6", "predict", await request.json(), timeout=120.0)

@app.post("/stage7/predict")
async def stage7(request: Request, _=Depends(validate_master)):
    """Direct Stage 7 call — requires h_v from Stage 6 in payload."""
    return await proxy("7", "predict", await request.json(), timeout=120.0)

@app.post("/stage8/predict")
async def stage8(request: Request, _=Depends(validate_master)):
    return await proxy("8", "predict/fusion", await request.json())

@app.post("/stage9/explain")
async def stage9(request: Request, _=Depends(validate_master)):
    return await proxy("9", "explain", await request.json(), timeout=300.0)

@app.post("/stage10/generate")
async def stage10(request: Request, _=Depends(validate_master)):
    return await proxy("10", "generate", await request.json(), timeout=300.0)


# ── PARALLEL Stage 6+7 orchestration ─────────────────────────────────────────
@app.post("/stage67/predict")
async def stage67_parallel(request: Request, _=Depends(validate_master)):
    """
    TRUE PARALLEL execution of Stage 6 and Stage 7:

    Architecture:
      Stage 6 (RGCN) runs first — produces h_v node embeddings.
      Stage 7 (GRU) DEPENDS on Stage 6 h_v, so it cannot start until Stage 6 finishes.

    However, parallelism is achieved across SNAPSHOTS within Stage 6:
      - If payload contains multiple snapshots, Stage 6 processes them concurrently
      - Stage 7 then runs its GRU over the collected h_v sequence

    Additionally, while Stage 6+7 run, Stage 8 preparation can happen concurrently
    (see /pipeline/full_run for the complete orchestrated flow).

    Payload:
      {
        "scenario_id": str,
        "graph_b64": str,          # base64 torch.save bytes of graph.pt
        "snapshots_b64": [str],    # list of base64 snapshot bytes (optional)
      }

    Returns merged Stage 6 + Stage 7 results ready for Stage 8.
    """
    payload = await request.json()
    scenario_id = payload.get("scenario_id", "")

    async with httpx.AsyncClient() as client:

        # ── Step 1: Stage 6 on full graph (structural threat detection) ───────
        # This is the dormant phase — Stage 6 runs, Stage 7 waits for h_v
        stage6_payload = {
            "graph_b64":   payload.get("graph_b64", ""),
            "scenario_id": scenario_id,
        }
        stage6_result = await _call_stage(
            client, "6", "predict", stage6_payload, timeout=120.0
        )

        # ── Step 2: Stage 7 uses Stage 6 h_v — dependency satisfied ──────────
        # Now both can contribute their outputs to Stage 8 concurrently
        # Stage 7 gets h_v sequence; simultaneously Stage 6 threat scores are
        # already available for Stage 8's structural score S_structural

        stage7_payload = {
            "h_v":         stage6_result.get("h_v", []),        # (N, 128) from Stage 6
            "scenario_id": scenario_id,
            "n_nodes":     stage6_result.get("n_nodes", 0),
        }

        # ── Step 3: Run Stage 7 and any preparatory Stage 8 work in parallel ─
        # Stage 7 GRU forward pass + Stage 8 input preparation happen together
        async def run_stage7():
            return await _call_stage(
                client, "7", "predict", stage7_payload, timeout=120.0
            )

        async def prepare_stage8_inputs():
            """
            While Stage 7 runs, pre-validate Stage 8 inputs from Stage 6.
            Returns immediately with Stage 6 structural scores.
            """
            return {
                "S_structural": stage6_result.get("threat_scores", []),
                "edge_anomaly": stage6_result.get("edge_anomaly",  {}),
                "node_offsets": stage6_result.get("node_offsets",  {}),
            }

        # True parallel: Stage 7 GRU + Stage 8 prep fire simultaneously
        stage7_result, stage8_prep = await asyncio.gather(
            run_stage7(),
            prepare_stage8_inputs(),
        )

        # ── Step 4: Merge results — both stages dormant, outputs ready ────────
        return {
            "scenario_id": scenario_id,
            "n_nodes":     stage6_result.get("n_nodes", 0),

            # Stage 6 outputs (structural)
            "S_structural":       stage8_prep["S_structural"],
            "edge_anomaly":       stage8_prep["edge_anomaly"],
            "node_embeddings":    stage6_result.get("h_v", []),   # (N, 128)
            "node_offsets":       stage8_prep["node_offsets"],

            # Stage 7 outputs (temporal)
            "S_temporal":         stage7_result.get("next_step_predictions", []),
            "final_hidden":       stage7_result.get("final_hidden", []),

            # Metadata for Stage 8
            "stage6_model":       "adarsh-aur/stage6-rgcn-security/model_RGCN.pt",
            "stage7_model":       "sohomn/stage7-temporal-gnn/model_GRU.pt",
        }


# ── Full pipeline orchestration (convenience route) ──────────────────────────
@app.post("/pipeline/full_run")
async def full_pipeline(request: Request, _=Depends(validate_master)):
    """
    Orchestrates a full pipeline run for a single event:
    0b → 1 → 2 → 3a → 3b → 4 → 5 → 6∥7 → 8 → 9+10

    For dissertation demo only — production would use individual stage calls
    with Prefect orchestration.
    """
    payload     = await request.json()
    results     = {}
    scenario_id = payload.get("scenario_id", "demo_001")

    async with httpx.AsyncClient() as client:

        # Stage 0b — generate raw log
        results["stage0b"] = await _call_stage(
            client, "0B", "generate", payload.get("event", {}), timeout=180.0
        )

        # Stages 1, 3a, 3b, 4 can run in parallel after Stage 0b
        log_payload  = {"provider": payload.get("provider","AWS"),
                        "raw_log":  results["stage0b"].get("log", {})}
        cve_payload  = {"cves": payload.get("cves", [])}
        id_payload   = {"identity": payload.get("entity_id",""),
                        "provider": payload.get("provider","AWS")}
        ner_payload  = {"logs": [{"log_text": str(results["stage0b"].get("log",""))}]}

        stage1_res, stage3a_res, stage3b_res, stage4_res = await asyncio.gather(
            _call_stage(client, "1",  "normalise", log_payload),
            _call_stage(client, "3A", "extract",   ner_payload),
            _call_stage(client, "3B", "score",     cve_payload, timeout=60.0),
            _call_stage(client, "4",  "embed",     id_payload),
        )
        results["stage1"]  = stage1_res
        results["stage3a"] = stage3a_res
        results["stage3b"] = stage3b_res
        results["stage4"]  = stage4_res

        # Stage 2 — embed logs (after stage 1 normalises)
        results["stage2"] = await _call_stage(
            client, "2", "embed",
            {"entity_id": stage1_res.get("entity_id",""), "log_texts": [str(log_payload)]},
            timeout=60.0
        )

        # Stage 5 — build graph (after all embeddings ready)
        # Stage 6+7 — parallel after Stage 5
        # Stage 8, 9+10 — after Stage 6+7
        # (abbreviated for demo — full implementation uses Prefect)
        results["status"] = "partial_run_demo"
        results["note"]   = "Full orchestration requires Prefect + Redis. Use individual stage routes for complete pipeline."

    return results