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
curl -X POST https://pipeline-gateway.onrender.com/stage3b/score \
  -H "X-API-Key: YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cves": [{"cve_id": "CVE-2024-9999", "description": "RCE in cloud VM agent via crafted packet"}]}'
```
