# import os, sys, torch
# from fastapi import FastAPI, Depends
# from pydantic import BaseModel
# from typing import List
# sys.path.append("..")
# from shared.auth import get_api_key_validator

# app      = FastAPI(title="Stage 3b - CVE Risk Scoring")
# validate = get_api_key_validator("3B")
# _model   = None


# def get_model():
#     global _model
#     if _model is None:
#         from transformers import AutoTokenizer, AutoModel
#         tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
#         mdl = AutoModel.from_pretrained("intfloat/e5-large-v2")
#         mdl.eval()
#         proj = torch.nn.Sequential(
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(512),
#             torch.nn.Linear(512, 128),
#         )
#         risk_head = torch.nn.Linear(128, 2)
#         _model = {"tok": tok, "mdl": mdl, "proj": proj, "risk": risk_head}
#     return _model


# @app.on_event("startup")
# async def startup():
#     get_model()


# class CVERequest(BaseModel):
#     cve_id:      str
#     description: str


# class CVEBatchRequest(BaseModel):
#     cves: List[CVERequest]


# @app.get("/health")
# async def health():
#     return {
#         "stage": "3b",
#         "status": "ok",
#         "model": "intfloat/e5-large-v2 + projection head",
#     }


# @app.post("/score")
# async def score(req: CVEBatchRequest, _=Depends(validate)):
#     m       = get_model()
#     results = []

#     for cve in req.cves:
#         text   = f"query: {cve.description}"
#         inputs = m["tok"](text, return_tensors="pt", truncation=True,
#                           max_length=512, padding=True)
#         with torch.no_grad():
#             out  = m["mdl"](**inputs)
#             mask = inputs["attention_mask"].unsqueeze(-1).float()
#             emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#             emb  = torch.nn.functional.normalize(emb, dim=-1)
#             z    = m["proj"](emb)
#             raw  = m["risk"](z)
#             risk = torch.sigmoid(raw[0][0]).item() * 10
#             expl = torch.sigmoid(raw[0][1]).item()

#         results.append({
#             "cve_id":       cve.cve_id,
#             "z_cve":        z[0].tolist(),
#             "risk_score":   round(risk, 4),
#             "exploit_prob": round(expl, 4),
#             "dim":          128,
#         })

#     return {"results": results}
