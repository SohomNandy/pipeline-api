"""
Stage 8 — Risk Fusion (Ensemble MLP)
Model: sohomn/risk-fusion-mlp-standard + sohomn/risk-fusion-mlp-fuzzy
Platform: Modal CPU
Input:  S_structural (Stage 6), S_temporal (Stage 7), risk_score (Stage 3b)
Output: final_threat_score + severity + ensemble audit trail
"""

import modal, os, sys, time, hmac, hashlib, asyncio, logging, importlib.util, threading
from collections import OrderedDict
from typing import Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# MODAL IMAGE + APP
# ══════════════════════════════════════════════════════════════════════════════

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "numpy>=1.26.0",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.23.0",
        "httpx>=0.27.0",
        "torch>=2.0.0",
    )
)

app = modal.App("stage8-risk-fusion", image=image)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR  (loaded once in __enter__, reused across requests)
# ══════════════════════════════════════════════════════════════════════════════

EMA_ALPHA            = float(os.environ.get("EMA_ALPHA",            "0.1"))
EMA_TRIP_RATIO       = float(os.environ.get("EMA_TRIP_RATIO",       "0.2"))
EMA_MIN_SAMPLES      = int(os.environ.get("EMA_MIN_SAMPLES",        "10"))
CERTAINTY_THRESHOLD  = float(os.environ.get("CERTAINTY_THRESHOLD",  "0.70"))
DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", "1.5"))

STD_REPO   = "sohomn/risk-fusion-mlp-standard"
FUZZY_REPO = "sohomn/risk-fusion-mlp-fuzzy"


@app.cls(
    cpu=2,
    memory=2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
class RiskFusionPredictor:

    def __enter__(self):
        from huggingface_hub import hf_hub_download

        def _load(repo_id: str, namespace: str):
            weights = hf_hub_download(repo_id, "risk_fusion_mlp.pt")
            cfg     = hf_hub_download(repo_id, "risk_fusion_config.json")
            src     = hf_hub_download(repo_id, "risk_fusion.py")
            spec    = importlib.util.spec_from_file_location(namespace, src)
            mod     = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = mod
            spec.loader.exec_module(mod)
            pipe = mod.RiskFusionPipeline.load(weights, cfg)
            print(f"✓ Loaded {repo_id}")
            return pipe

        self.std_pipe   = _load(STD_REPO,   "rf_standard")
        self.fuzzy_pipe = _load(FUZZY_REPO, "rf_fuzzy")
        self.ema        = {"value": None, "n": 0}

    @modal.method()
    def predict(
        self,
        S_structural:     float,
        S_temporal:       float,
        cvss:             float,
        exploitability:   float,
        impact:           float,
        identity_anomaly: float,
        node_id:          str,
    ) -> dict:
        kwargs = dict(
            structural=S_structural, temporal=S_temporal,
            cvss=cvss, exploitability=exploitability,
            impact=impact, identity_anomaly=identity_anomaly,
        )
        std_out   = self.std_pipe.predict(**kwargs)
        fuzzy_out = self.fuzzy_pipe.predict(**kwargs)

        std_score   = float(std_out["Final Risk Score"])
        fuzzy_score = float(fuzzy_out["Final Risk Score"])
        certainty   = float(fuzzy_out["Certainty"])
        defuzzified = float(fuzzy_out["Defuzzified"])
        divergence  = abs(std_score - fuzzy_score)

        if divergence > DIVERGENCE_THRESHOLD and certainty < CERTAINTY_THRESHOLD:
            raw        = max(std_score, fuzzy_score)
            blend_mode = "conservative_max"
            w_std = w_fuzzy = None
        else:
            w_fuzzy    = certainty
            w_std      = 1.0 - certainty
            raw        = w_std * std_score + w_fuzzy * defuzzified
            blend_mode = "certainty_weighted"

        final_01   = round(float(np.clip(raw, 0.0, 10.0)) / 10.0, 4)
        cb_tripped = self._ema_check(final_01, node_id)

        return {
            "final_threat_score": final_01,
            "severity":           self._severity(final_01),
            "alert_priority":     fuzzy_out["Alert Priority"],
            "certainty":          round(certainty, 4),
            "feature_weights":    std_out["Feature Weights"],
            "fuzzy_memberships":  fuzzy_out["Fuzzy Memberships"],
            "cb_tripped":         cb_tripped,
            "ensemble_meta": {
                "std_score":       round(std_score, 4),
                "fuzzy_score":     round(fuzzy_score, 4),
                "defuzzified":     round(defuzzified, 4),
                "divergence":      round(divergence, 4),
                "divergence_flag": divergence > DIVERGENCE_THRESHOLD,
                "blend_mode":      blend_mode,
                "w_std":           round(w_std,   4) if w_std   is not None else None,
                "w_fuzzy":         round(w_fuzzy, 4) if w_fuzzy is not None else None,
            },
        }

    def _severity(self, score_01: float) -> str:
        if score_01 >= 0.90: return "Critical"
        if score_01 >= 0.75: return "High"
        if score_01 >= 0.50: return "Medium"
        return "Low"

    def _ema_check(self, score: float, node_id: str) -> bool:
        if self.ema["value"] is None:
            self.ema["value"] = score
            self.ema["n"] = 1
            return False
        self.ema["value"] = EMA_ALPHA * score + (1 - EMA_ALPHA) * self.ema["value"]
        self.ema["n"] += 1
        if self.ema["n"] < EMA_MIN_SAMPLES:
            return False
        if score / (self.ema["value"] + 1e-9) < EMA_TRIP_RATIO:
            print(f"EMA CB tripped | node={node_id} score={score:.4f} ema={self.ema['value']:.4f}")
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI WRAPPER  (mirrors Stage 6 pattern exactly)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    cpu=2,
    memory=2048,
    secrets=[modal.Secret.from_name("siem-pipeline-secrets")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import secrets as _secrets
    import httpx
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel, Field, validator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    log = logging.getLogger("stage8")

    DISCORD_WEBHOOK  = os.environ.get("DISCORD_WEBHOOK_URL", "")
    TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN",  "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",    "")
    NONCE_WINDOW     = int(os.environ.get("NONCE_WINDOW_SECS", "60"))

    # ── API key validation (same pattern as Stage 6) ──────────────────────────
    API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

    def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get("STAGE_8_API_KEY", "")
        if not api_key or not _secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    # ── hash chain ────────────────────────────────────────────────────────────
    def _hash_chain(node_id: str, score: float, stage_label: str) -> str:
        return hashlib.sha256(f"{node_id}:{score:.6f}:{stage_label}".encode()).hexdigest()

    # ── nonce store ───────────────────────────────────────────────────────────
    class _NonceStore:
        def __init__(self, window_secs: int):
            self._window = window_secs
            self._store: OrderedDict[str, float] = OrderedDict()
            self._lock = threading.Lock()

        def consume(self, nonce: str) -> bool:
            now = time.time()
            with self._lock:
                cutoff = now - self._window
                while self._store:
                    k, ts = next(iter(self._store.items()))
                    if ts < cutoff:
                        self._store.popitem(last=False)
                    else:
                        break
                if nonce in self._store:
                    return False
                self._store[nonce] = now
                return True

    nonce_store = _NonceStore(NONCE_WINDOW)

    # ── alerts ────────────────────────────────────────────────────────────────
    _TIMEOUT = httpx.Timeout(5.0)

    async def _send_siren(message: str):
        tasks = []
        if DISCORD_WEBHOOK:
            async def _discord():
                async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                    r = await c.post(DISCORD_WEBHOOK, json={"content": message, "username": "Trinetra Stage 8"})
                    if r.status_code not in (200, 204):
                        log.error("Discord %s %s", r.status_code, r.text)
            tasks.append(_discord())
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            async def _telegram():
                async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                    r = await c.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
                    )
                    if r.status_code != 200:
                        log.error("Telegram %s %s", r.status_code, r.text)
            tasks.append(_telegram())
        if tasks:
            for res in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(res, Exception):
                    log.error("Alert error: %s", res)
        else:
            log.warning("SIREN | %s", message)

    # ── schemas ───────────────────────────────────────────────────────────────
    class PipelineMeta(BaseModel):
        edge_id:      str
        scenario_id:  str
        t:            int
        malicious:    Optional[int] = None
        attack_phase: Optional[str] = None
        stage_6_hash: str
        stage_7_hash: str
        nonce:        Optional[str] = None

    class FusionRequest(BaseModel):
        node_id:          str
        S_structural:     float = Field(..., ge=0.0, le=1.0)
        S_temporal:       float = Field(..., ge=0.0, le=1.0)
        risk_score:       float = Field(..., ge=0.0, le=10.0)
        exploitability:   float = Field(0.0, ge=0.0, le=3.9)
        impact:           float = Field(0.0, ge=0.0, le=6.0)
        identity_anomaly: float = Field(0.0, ge=0.0, le=1.0)
        pipeline_meta:    PipelineMeta

        @validator("S_structural", "S_temporal", "identity_anomaly", pre=True)
        def clamp_01(cls, v):
            return float(np.clip(v, 0.0, 1.0))

    # ── app + predictor ───────────────────────────────────────────────────────
    web       = FastAPI(title="Stage 8 — Risk Fusion")
    predictor = RiskFusionPredictor()

    @web.get("/health")
    async def health():
        return {
            "stage":  "8",
            "status": "ok",
            "models": [STD_REPO, FUZZY_REPO],
        }

    @web.post("/predict/fusion")
    async def predict_fusion(payload: FusionRequest, _=Depends(validate)):
        t0 = time.perf_counter()

        # 1. Nonce / replay protection
        if payload.pipeline_meta.nonce:
            if not nonce_store.consume(payload.pipeline_meta.nonce):
                log.error("Replay | nonce=%s node=%s", payload.pipeline_meta.nonce, payload.node_id)
                raise HTTPException(status_code=400, detail="Replay detected: nonce already used")

        # 2. Hash chain — verify stage 6 + 7 before touching scores
        if payload.pipeline_meta.stage_6_hash != _hash_chain(payload.node_id, payload.S_structural, "stage_6"):
            asyncio.create_task(_send_siren(
                f"❌ stage_6_hash mismatch | node=`{payload.node_id}` scenario=`{payload.pipeline_meta.scenario_id}`"
            ))
            raise HTTPException(status_code=400, detail="stage_6_hash integrity check failed")

        if payload.pipeline_meta.stage_7_hash != _hash_chain(payload.node_id, payload.S_temporal, "stage_7"):
            asyncio.create_task(_send_siren(
                f"❌ stage_7_hash mismatch | node=`{payload.node_id}` scenario=`{payload.pipeline_meta.scenario_id}`"
            ))
            raise HTTPException(status_code=400, detail="stage_7_hash integrity check failed")

        # 3. Ensemble inference
        result = predictor.predict.remote(
            S_structural     = payload.S_structural,
            S_temporal       = payload.S_temporal,
            cvss             = payload.risk_score,
            exploitability   = payload.exploitability,
            impact           = payload.impact,
            identity_anomaly = payload.identity_anomaly,
            node_id          = payload.node_id,
        )

        # 4. Siren alerts (fire-and-forget)
        if result["cb_tripped"]:
            asyncio.create_task(_send_siren(
                f"⚡ EMA CB tripped | node=`{payload.node_id}` "
                f"score={result['final_threat_score']:.4f} "
                f"scenario=`{payload.pipeline_meta.scenario_id}`"
            ))
        if result["severity"] in ("Critical", "High"):
            asyncio.create_task(_send_siren(
                f"🚨 [{result['severity']}] node=`{payload.node_id}` "
                f"score={result['final_threat_score']:.4f} priority={result['alert_priority']} "
                f"scenario=`{payload.pipeline_meta.scenario_id}` t={payload.pipeline_meta.t}"
            ))

        # 5. Stage 8 hash for Stage 9 downstream
        stage_8_hash = _hash_chain(payload.node_id, result["final_threat_score"], "stage_8")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "node=%s score=%.4f severity=%s blend=%s divergence=%.4f certainty=%.4f cb=%s latency=%sms",
            payload.node_id, result["final_threat_score"], result["severity"],
            result["ensemble_meta"]["blend_mode"], result["ensemble_meta"]["divergence"],
            result["certainty"], result["cb_tripped"], elapsed_ms,
        )

        return {
            "node_id":            payload.node_id,
            "final_threat_score": result["final_threat_score"],
            "severity":           result["severity"],
            "alert_priority":     result["alert_priority"],
            "certainty":          result["certainty"],
            "feature_weights":    result["feature_weights"],
            "fuzzy_memberships":  result["fuzzy_memberships"],
            "ensemble_meta": {
                **result["ensemble_meta"],
                "cb_tripped": result["cb_tripped"],
                "latency_ms": elapsed_ms,
            },
            "pipeline_meta": {
                **payload.pipeline_meta.dict(),
                "stage_8_hash": stage_8_hash,
            },
        }

    return web