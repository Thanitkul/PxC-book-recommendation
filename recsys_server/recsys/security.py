import hmac
import hashlib
from config import HMAC_SECRET


def verify_hmac_signature(method: str, path: str, timestamp: str, raw_body: bytes, client_signature: str) -> bool:
    # Must match the stringified payload from JS
    body_str = raw_body.decode("utf-8")
    payload = f"{method}\n{path}\n{timestamp}\n{body_str}"

    computed_signature = hmac.new(
        key=HMAC_SECRET.encode(),
        msg=payload.encode(),
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed_signature, client_signature)
