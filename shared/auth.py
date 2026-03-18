import os, secrets, hashlib
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_validator(stage: str):
    """Returns a FastAPI dependency that validates X-API-Key for the given stage."""
    env_var = f"STAGE_{stage}_API_KEY"

    async def validate(api_key: str = Security(API_KEY_HEADER)):
        expected = os.environ.get(env_var, "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail=f"API key not configured for stage {stage}"
            )
        if not api_key or not secrets.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(expected.encode()).hexdigest(),
        ):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    return validate
