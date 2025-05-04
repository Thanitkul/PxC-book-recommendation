from fastapi import APIRouter, HTTPException, Request, status
from recsys.security import verify_hmac_signature
from recsys.logic.inference import recommend_books
from pydantic import BaseModel
import json

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/recommend")
async def recommend(request: Request):
    # Extract headers
    signature = request.headers.get("X-Signature")
    timestamp = request.headers.get("X-Timestamp")
    if not signature or not timestamp:
        raise HTTPException(status_code=400, detail="Missing signature or timestamp")

    # Read raw body for HMAC
    raw_body = await request.body()

    # Verify HMAC
    if not verify_hmac_signature("POST", "/api/recsys/recommend", timestamp, raw_body, signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid HMAC signature")

    # Parse JSON payload
    try:
        body = json.loads(raw_body.decode("utf-8"))
        user_id = body["user_id"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Return recommendations
    book_ids = await recommend_books(user_id)
    return {"recommendations": book_ids}
