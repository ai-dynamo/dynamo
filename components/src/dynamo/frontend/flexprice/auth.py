"""JWT authentication helpers.

Validates a Bearer JWT signed with HMAC-SHA256/384/512, extracts the
``org_uuid`` claim, and optionally enforces an org allowlist.  Implemented
using the Python standard library only (no PyJWT dependency).
"""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class AuthError(Exception):
    """Raised when a request fails authentication or authorisation."""

    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class AuthCtx:
    """Decoded claims from a validated JWT."""

    user_uuid: str
    org_uuid: str
    token_uuid: str = ""


# ---------------------------------------------------------------------------
# Internal JWT helpers
# ---------------------------------------------------------------------------

def _b64url_decode(data: str) -> bytes:
    """Decode a base64url-encoded string (no padding required)."""
    rem = len(data) % 4
    if rem:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data)


_HASH_FN = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}


def _verify_jwt(token: str, secret_keys: List[str]) -> Dict:
    """Verify an HMAC-signed JWT and return its claims.

    Tries every key in *secret_keys* to support key rotation.  Raises
    :class:`AuthError` for any validation failure.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("invalid token format")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()

    try:
        header = json.loads(_b64url_decode(header_b64))
    except Exception:
        raise AuthError("malformed token header")

    alg = header.get("alg", "")
    hash_fn = _HASH_FN.get(alg)
    if hash_fn is None:
        raise AuthError(f"unsupported signing algorithm: {alg!r}")

    try:
        expected_sig = _b64url_decode(sig_b64)
    except Exception:
        raise AuthError("malformed token signature")

    verified = False
    for secret in secret_keys:
        computed = hmac.new(secret.encode(), signing_input, hash_fn).digest()
        if hmac.compare_digest(computed, expected_sig):
            verified = True
            break

    if not verified:
        raise AuthError("invalid token signature")

    try:
        claims: Dict = json.loads(_b64url_decode(payload_b64))
    except Exception:
        raise AuthError("malformed token payload")

    exp = claims.get("exp")
    if exp is not None and exp < time.time():
        raise AuthError("token expired")

    return claims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def authenticate(
    auth_header: str,
    secret_keys: List[str],
    valid_orgs: Optional[List[str]] = None,
) -> AuthCtx:
    """Authenticate a ``Authorization: Bearer <jwt>`` header.

    Args:
        auth_header: Raw value of the ``Authorization`` HTTP header.
        secret_keys: One or more HMAC signing secrets (tried in order to
            support key rotation).
        valid_orgs: If non-empty, only orgs in this list are allowed.

    Returns:
        :class:`AuthCtx` with decoded identity fields.

    Raises:
        :class:`AuthError` (status 401) on any authentication failure.
    """
    if not auth_header:
        raise AuthError("Authorization header is required")

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be: Bearer <token>")

    token = parts[1].strip()
    claims = _verify_jwt(token, secret_keys)

    user_uuid = claims.get("uuid")
    org_uuid = claims.get("org_uuid")

    if not user_uuid:
        raise AuthError("token is missing the 'uuid' claim")
    if not org_uuid:
        raise AuthError("token is missing the 'org_uuid' claim")

    if valid_orgs and org_uuid not in valid_orgs:
        raise AuthError(
            f"organization {org_uuid!r} is not permitted to access this endpoint"
        )

    return AuthCtx(
        user_uuid=str(user_uuid),
        org_uuid=str(org_uuid),
        token_uuid=str(claims.get("token_uuid", "")),
    )
