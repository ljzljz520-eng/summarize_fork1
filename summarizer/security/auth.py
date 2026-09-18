"""API-token and OIDC authentication with scope enforcement.

Two credential mechanisms are supported for ``authenticated-server`` mode:

* static API tokens configured as SHA-256 hashes (constant-time comparison);
* OIDC bearer JWTs verified against the issuer's JWKS (RS256/ES256).

Credentials are never logged; failure responses never echo submitted secrets.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status

from ..exceptions import ConfigurationError
from .netpolicy import PURPOSE_OIDC
from .settings import (
    ALL_KNOWN_SCOPES,
    OidcSettings,
    ServerSettings,
    hash_token,
)

logger = logging.getLogger(__name__)


def _parse_claim_scopes(raw: Any) -> FrozenSet[str]:
    """Parse scopes carried in a JWT claim, silently ignoring unknown ones.

    Claims are issuer-controlled free-form data (unlike server token config);
    an unrecognized scope string must never turn authentication into a 500.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
    elif isinstance(raw, (list, tuple, set, frozenset)):
        tokens = []
        for item in raw:
            tokens.extend(str(item).replace(",", " ").split())
    else:
        return frozenset()
    known = frozenset(ALL_KNOWN_SCOPES)
    return frozenset(token.strip() for token in tokens if token.strip()) & known


def _safe_log_subject(subject: str) -> str:
    """Neutralize CR/LF so an externally controlled subject cannot forge logs."""
    return str(subject).replace("\r", "\\r").replace("\n", "\\n")[:200]

_JWT_SHAPE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$")
_JWKS_TTL = 3600.0


@dataclass(frozen=True)
class Identity:
    subject: str
    method: str  # "token" | "oidc"
    scopes: FrozenSet[str]
    providers: Optional[FrozenSet[str]] = None
    claims: Optional[Dict[str, Any]] = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def provider_allowed(self, provider_id: Optional[str]) -> bool:
        if self.providers is None:
            return True
        return provider_id in self.providers


class AuthenticationError(Exception):
    """Raised internally for 401 failures (invalid/missing credentials)."""


class AuthorizationError(Exception):
    """Raised internally for 403 failures (valid identity, insufficient scope)."""


# ── Static tokens ───────────────────────────────────────────────────────────


def authenticate_token(raw_token: str, settings: ServerSettings) -> Identity:
    candidate = hash_token(raw_token.strip())
    matched = None
    for entry in settings.tokens:
        # compare_digest over fixed-length hex digests; hmac accepts str too
        # but bytes keep semantics explicit.
        if _compare_digest(candidate, entry.token_hash):
            matched = entry
            break
    if matched is None:
        raise AuthenticationError("invalid API token")
    return Identity(
        subject=matched.id,
        method="token",
        scopes=matched.scopes,
        providers=matched.providers,
    )


def _compare_digest(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ── OIDC ────────────────────────────────────────────────────────────────────


def _import_jose():
    try:
        from jose import jwt  # noqa: F401
        from jose.exceptions import JWTError  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError(
            "OIDC authentication requires the 'python-jose[cryptography]' "
            "package. Install the server extra: "
            "`pip install summarizer[server]`, or use a static API token "
            "(SUMMARIZER_API_TOKEN)."
        ) from exc
    import jose.jwt as jwt_module

    return jwt_module


def identity_from_claims(
    claims: Dict[str, Any], oidc: OidcSettings
) -> Identity:
    scopes = _parse_claim_scopes(claims.get(oidc.scopes_claim))
    providers = None
    if oidc.providers_claim:
        raw_providers = claims.get(oidc.providers_claim)
        if raw_providers is not None:
            if isinstance(raw_providers, str):
                providers = frozenset(raw_providers.split())
            else:
                providers = frozenset(str(item) for item in raw_providers)
    subject = str(claims.get("sub") or "oidc-anonymous")
    return Identity(
        subject=subject,
        method="oidc",
        scopes=scopes,
        providers=providers,
        claims=claims,
    )


class OidcVerifier:
    """Verifies RS256/ES256 JWTs against an issuer JWKS endpoint."""

    def __init__(
        self,
        oidc: OidcSettings,
        *,
        jwks_fetcher: Optional[Callable[[], Dict[str, Any]]] = None,
        policy: Any = None,
    ):
        self.oidc = oidc
        self._jwt = _import_jose()
        self._lock = threading.Lock()
        self._cached_jwks: Optional[Tuple[float, Dict[str, Any]]] = None
        self._jwks_fetcher = jwks_fetcher
        self._policy = policy

    def _fetch_jwks(self) -> Dict[str, Any]:
        if self._jwks_fetcher is not None:
            return self._jwks_fetcher()
        from .httpguards import guarded_request

        jwks_uri = self.oidc.jwks_uri
        if not jwks_uri:
            discovery_url = (
                f"{self.oidc.issuer.rstrip('/')}/.well-known/openid-configuration"
            )
            from .httpguards import preflight_url

            preflight_url(self._policy, discovery_url, PURPOSE_OIDC)
            response = guarded_request(
                "get",
                discovery_url,
                policy=self._policy,
                purpose=PURPOSE_OIDC,
                timeout=15,
            )
            response.raise_for_status()
            metadata = response.json()
            jwks_uri = metadata.get("jwks_uri")
            if not jwks_uri:
                raise ConfigurationError(
                    f"OIDC discovery document at {discovery_url} has no jwks_uri."
                )
        from .httpguards import preflight_url

        preflight_url(self._policy, jwks_uri, PURPOSE_OIDC)
        response = guarded_request(
            "get",
            jwks_uri,
            policy=self._policy,
            purpose=PURPOSE_OIDC,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _get_jwks(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if (
                not force_refresh
                and self._cached_jwks is not None
                and now - self._cached_jwks[0] < _JWKS_TTL
            ):
                return self._cached_jwks[1]
        jwks = self._fetch_jwks()
        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise ConfigurationError("OIDC JWKS endpoint returned an invalid document.")
        with self._lock:
            self._cached_jwks = (now, jwks)
        return jwks

    def _key_for(self, token: str, jwks: Dict[str, Any]) -> Dict[str, Any]:
        try:
            unverified_header = self._jwt.get_unverified_header(token)
        except Exception as exc:
            raise AuthenticationError("malformed JWT header") from exc
        alg = unverified_header.get("alg", "")
        if alg not in self.oidc.algorithms or alg.lower() == "none":
            raise AuthenticationError(f"unsupported JWT algorithm: {alg!r}")
        kid = unverified_header.get("kid")
        for key in jwks.get("keys", []):
            if kid is not None and key.get("kid") != kid:
                continue
            return key
        raise AuthenticationError("no matching JWKS key for token")

    def verify(self, token: str) -> Identity:
        jwks = self._get_jwks()
        try:
            key = self._key_for(token, jwks)
        except AuthenticationError:
            # Key may have rotated: refresh once and retry.
            jwks = self._get_jwks(force_refresh=True)
            key = self._key_for(token, jwks)
        try:
            claims = self._jwt.decode(
                token,
                key,
                algorithms=list(self.oidc.algorithms),
                audience=self.oidc.audience,
                issuer=self.oidc.issuer,
                options={
                    "require_iss": True,
                    "require_sub": False,
                    # A token without expiry or audience must never be accepted:
                    # it could outlive revocation or target another service.
                    "require_exp": True,
                    "require_aud": True,
                    "require_iat": True,
                },
            )
        except Exception as exc:
            name = type(exc).__name__
            raise AuthenticationError(f"JWT verification failed: {name}") from exc
        return identity_from_claims(claims, self.oidc)


def looks_like_jwt(token: str) -> bool:
    return bool(_JWT_SHAPE.match(token.strip()))


# ── FastAPI dependencies ────────────────────────────────────────────────────


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )
    if not header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        # Do not echo the malformed header value.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header; expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1].strip()


def _matches_static_token(raw: str, settings: ServerSettings) -> bool:
    """Prefer a static token even when its random content has JWT-like shape."""
    candidate = hash_token(raw)
    return any(_compare_digest(candidate, entry.token_hash) for entry in settings.tokens)


class AuthGuard:
    """Per-app authentication/scope dependencies for one ServerSettings."""

    def __init__(
        self,
        settings: ServerSettings,
        verifier: Optional[OidcVerifier] = None,
    ):
        self.settings = settings
        self.verifier = verifier

    def identity(self, request: Request) -> Identity:
        raw = _extract_bearer(request)
        try:
            try:
                if (
                    self.settings.oidc is not None
                    and looks_like_jwt(raw)
                    and not _matches_static_token(raw, self.settings)
                ):
                    active_verifier = self.verifier or OidcVerifier(self.settings.oidc)
                    result = active_verifier.verify(raw)
                else:
                    result = authenticate_token(raw, self.settings)
            except AuthenticationError:
                raise
            except Exception as exc:
                # JWKS fetch/network/config/policy failures must surface as a
                # generic 401 (never a 500 and never with internal details).
                logger.warning(
                    "OIDC verification error: %s", type(exc).__name__
                )
                raise AuthenticationError(
                    f"JWT verification failed: {type(exc).__name__}"
                ) from exc
        except AuthenticationError as exc:
            logger.warning("Authentication failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        except AuthorizationError as exc:
            logger.warning("Authorization failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            ) from exc
        return result

    def require_scope(self, *required_scopes: str):
        """Return a FastAPI dependency enforcing ALL listed scopes."""

        def dependency(identity: Identity = Depends(self.identity)) -> Identity:
            missing = [
                scope for scope in required_scopes if not identity.has_scope(scope)
            ]
            if missing:
                logger.warning(
                    "Identity %r missing scope(s): %s",
                    _safe_log_subject(identity.subject),
                    ", ".join(missing),
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope(s): {', '.join(missing)}.",
                )
            return identity

        return dependency


def make_auth_guard(
    settings: ServerSettings,
    verifier: Optional[OidcVerifier] = None,
) -> AuthGuard:
    """Build the per-app authentication guard."""
    return AuthGuard(settings, verifier)
