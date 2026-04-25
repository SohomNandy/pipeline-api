"""
Stage 3a — Vulnerability NER
Platform: Render CPU (free tier, 512MB) ✓
Model: NONE — rule-based regex + keyword extraction
Memory: ~45MB total. No HF_TOKEN needed. No cold start. Always ready.

Extracts the same 8 entity types as CodeBERT:
  SOFTWARE, VERSION, ERROR, EXPLOIT, IP, PORT, USER, PATH
CVE IDs extracted via regex — 100% recall on standard CVE-YYYY-NNNNN format.
Response schema identical to CodeBERT version — zero downstream changes needed.
"""

import os, re, secrets as _secrets, hashlib, logging
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("stage3a")

API_KEY = os.environ.get("STAGE_3A_API_KEY", "")
PORT    = int(os.environ.get("PORT", "8000"))

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not _secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(),
        hashlib.sha256(API_KEY.encode()).hexdigest(),
    ):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# ══════════════════════════════════════════════════════════════════════════════
# COMPILED PATTERNS — once at import, reused for every request
# ══════════════════════════════════════════════════════════════════════════════

_RE_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

_RE_IP  = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

_RE_PORT = re.compile(r"(?:port\s+|:)(\d{2,5})\b", re.IGNORECASE)

_RE_PATH = re.compile(r"(?:/[\w.\-_/]{3,}|[A-Za-z]:\\[\w.\-_\\]{3,})")

_RE_SW_VER = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_.\-]{1,30})[\s/\-v]+(\d+(?:\.\d+){1,4})\b",
    re.IGNORECASE,
)

_RE_VERSION = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)*)\b", re.IGNORECASE)

_RE_USER = re.compile(
    r"(?:userName|principalId|principalEmail|userIdentity|user)\s*[=:\"\':\s]+"
    r"([A-Za-z0-9_.@\-]{3,64})",
    re.IGNORECASE,
)

_KNOWN_SW = frozenset({
    "apache","nginx","openssh","openssl","log4j","log4shell","struts","spring",
    "kubernetes","docker","containerd","python","java","node","nodejs","php",
    "ruby","perl","mysql","postgres","postgresql","mongodb","redis",
    "elasticsearch","chrome","firefox","curl","wget","bash","sudo","linux",
    "windows","ubuntu","debian","centos","rhel","terraform","ansible","jenkins",
    "gitlab","github","iis","tomcat","wordpress","drupal","joomla","ssh","ftp",
    "smb","rdp","vnc","ldap","kerberos","ntlm","mimikatz","metasploit",
})

_EXPLOIT_KW = frozenset({
    "exploit","exploitation","vulnerability","vuln","rce","lfi","rfi","sqli",
    "xss","overflow","injection","privilege escalation","lateral movement",
    "exfiltration","backdoor","shellcode","payload","reverse shell","metasploit",
    "mimikatz","log4shell","heartbleed","eternalblue","zerologon","printnightmare",
    "proxylogon","proxyshell","log4j","deserialization","path traversal",
    "directory traversal","command injection","code execution",
})


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract(text: str) -> dict:
    # ── CVE IDs (ERROR type — matches CodeBERT label convention) ─────────────
    cve_ids = list({m.upper() for m in _RE_CVE.findall(text)})

    # ── IPs ───────────────────────────────────────────────────────────────────
    ips = list(set(_RE_IP.findall(text)))

    # ── Ports ────────────────────────────────────────────────────────────────
    ports = list({p for p in _RE_PORT.findall(text) if 1 <= int(p) <= 65535})

    # ── Paths ────────────────────────────────────────────────────────────────
    paths = list({p for p in _RE_PATH.findall(text) if len(p) > 4})

    # ── Software + versions ───────────────────────────────────────────────────
    software, versions = [], []
    for m in _RE_SW_VER.finditer(text):
        sw, ver = m.group(1), m.group(2)
        if sw.lower() in _KNOWN_SW or (sw[0].isupper() and len(sw) > 2):
            if sw not in software:
                software.append(sw)
        if ver not in versions:
            versions.append(ver)

    for m in _RE_VERSION.finditer(text):
        v = m.group(1)
        if v not in versions:
            versions.append(v)

    # ── Users ─────────────────────────────────────────────────────────────────
    users = list(dict.fromkeys(m for m in _RE_USER.findall(text) if m))

    # ── Exploit keywords ─────────────────────────────────────────────────────
    tl = text.lower()
    exploits = [kw for kw in _EXPLOIT_KW if kw in tl]

    # ── Flat entities list (identical schema to CodeBERT output) ─────────────
    entities = []
    for x in cve_ids:  entities.append({"text": x, "type": "ERROR"})
    for x in software: entities.append({"text": x, "type": "SOFTWARE"})
    for x in versions: entities.append({"text": x, "type": "VERSION"})
    for x in ips:      entities.append({"text": x, "type": "IP"})
    for x in ports:    entities.append({"text": x, "type": "PORT"})
    for x in paths:    entities.append({"text": x, "type": "PATH"})
    for x in users:    entities.append({"text": x, "type": "USER"})
    for x in exploits: entities.append({"text": x, "type": "EXPLOIT"})

    return {
        "entities": entities, "cve_ids": cve_ids,
        "software": software, "versions": versions,
        "ips": ips,           "ports":   ports,
        "paths": paths,       "users":   users,
        "exploits": exploits,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Stage 3a — Vulnerability NER (rule-based)", version="4.0.0")

class ExtractRequest(BaseModel):
    log_text:    str
    edge_id:     Optional[str] = ""
    scenario_id: Optional[str] = ""
    t:           Optional[int] = 0

class ExtractBatchRequest(BaseModel):
    logs: List[ExtractRequest]


@app.get("/health")
def health():
    # Always ready — no model to load
    return {
        "stage":       "3a",
        "status":      "ok",
        "model_ready": True,
        "model":       "rule-based (regex + keywords, no weights)",
    }


@app.post("/extract")
def extract(req: ExtractBatchRequest, _=Depends(validate)):
    results = []
    for r in req.logs:
        try:
            e = _extract(r.log_text)
            results.append({
                "edge_id":    r.edge_id,  "scenario_id": r.scenario_id, "t": r.t,
                "log_text":   r.log_text[:200],
                "entities":   e["entities"],  "cve_ids":  e["cve_ids"],
                "software":   e["software"],  "versions": e["versions"],
                "exploits":   e["exploits"],  "ips":      e["ips"],
                "users":      e["users"],     "ports":    e["ports"],
                "paths":      e["paths"],     "errors":   [],
                "n_entities": len(e["entities"]),
            })
        except Exception as ex:
            log.warning(f"NER error: {ex}")
            results.append({
                "edge_id": r.edge_id, "scenario_id": r.scenario_id, "t": r.t,
                "log_text": r.log_text[:200], "error": str(ex),
                "entities":[], "cve_ids":[], "software":[], "versions":[],
                "exploits":[], "ips":[], "users":[], "ports":[], "paths":[],
                "errors":[], "n_entities": 0,
            })
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")