"""Deployment-mode and server security settings.

Three deployment modes are supported:

* ``compatibility``        - standalone CLI / Streamlit; no network restrictions
                             and no authentication (fixed for non-server usage).
* ``local-trusted``        - single-user API server; defaults to loopback bind;
                             behaves like the historical ``summarizer serve``.
* ``authenticated-server`` - multi-client deployment; every endpoint except
                             ``/health`` requires an API token or OIDC identity,
                             and the strict outbound policy is installed.

Settings are sourced with the precedence  CLI arguments > environment variables
> YAML (``server:`` / ``outbound:`` sections) > defaults.
"""

import hashlib
import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..config_file import load_config_file
from ..exceptions import ConfigurationError

MODE_COMPATIBILITY = "compatibility"
MODE_LOCAL_TRUSTED = "local-trusted"
MODE_AUTHENTICATED_SERVER = "authenticated-server"
VALID_MODES = (
    MODE_COMPATIBILITY,
    MODE_LOCAL_TRUSTED,
    MODE_AUTHENTICATED_SERVER,
)
SERVER_MODES = (MODE_LOCAL_TRUSTED, MODE_AUTHENTICATED_SERVER)

# ── Scopes ──────────────────────────────────────────────────────────────────

SCOPE_READ = "summarize:read"
SCOPE_WRITE = "summarize:write"
SCOPE_UPLOAD = "summarize:upload"
SCOPE_ADMIN_CONFIG = "admin:config"
SCOPE_SOURCES_LOCAL = "sources:local"
ALL_KNOWN_SCOPES: Tuple[str, ...] = (
    SCOPE_READ,
    SCOPE_WRITE,
    SCOPE_UPLOAD,
    SCOPE_ADMIN_CONFIG,
    SCOPE_SOURCES_LOCAL,
)
# Static tokens configured without an explicit scope list default to full access.
DEFAULT_TOKEN_SCOPES: FrozenSet[str] = frozenset(ALL_KNOWN_SCOPES)

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SHA256_PREFIX = "sha256:"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _env_list(name: str, separator: str = ",") -> Optional[List[str]]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(separator)]
    return [item for item in items if item]


def hash_token(raw_token: str) -> str:
    """Return the SHA-256 hex digest used for constant-time token lookup."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def normalize_stored_token(value: str) -> str:
    """Normalize a configured token to its stored form (SHA-256 hex digest).

    A literal ``sha256:<hex>`` value is accepted as-is; any other value is
    treated as a plaintext token and hashed at startup.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("Auth tokens must be non-empty strings.")
    value = value.strip()
    if value.startswith(_SHA256_PREFIX):
        digest = value[len(_SHA256_PREFIX):].strip().lower()
        try:
            int(digest, 16)
        except ValueError:
            raise ConfigurationError(
                "token_hash must be 'sha256:' followed by a hexadecimal digest."
            )
        if len(digest) != 64:
            raise ConfigurationError(
                "token_hash must contain a 64-character SHA-256 hex digest."
            )
        return digest
    return hash_token(value)


def parse_scopes(value: Any) -> FrozenSet[str]:
    """Parse scopes from a YAML list or a comma/whitespace separated string."""
    if value is None:
        return frozenset()
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set, frozenset)):
        tokens = []
        for item in value:
            tokens.extend(str(item).replace(",", " ").split())
    else:
        raise ConfigurationError(f"Unsupported scopes value: {value!r}")
    scopes = frozenset(token.strip() for token in tokens if token.strip())
    unknown = sorted(scope for scope in scopes if scope not in ALL_KNOWN_SCOPES)
    if unknown:
        raise ConfigurationError(
            f"Unknown scope(s): {', '.join(unknown)}. "
            f"Valid scopes: {', '.join(ALL_KNOWN_SCOPES)}"
        )
    return scopes


# ── Setting data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenEntry:
    id: str
    token_hash: str
    scopes: FrozenSet[str]
    providers: Optional[FrozenSet[str]] = None


@dataclass(frozen=True)
class OidcSettings:
    issuer: str
    audience: str
    jwks_uri: Optional[str] = None
    algorithms: Tuple[str, ...] = ("RS256", "ES256")
    scopes_claim: str = "scope"
    providers_claim: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.issuer and self.audience)


@dataclass(frozen=True)
class ServerSettings:
    mode: str
    cors_origins: Optional[Tuple[str, ...]] = None
    tokens: Tuple[TokenEntry, ...] = ()
    oidc: Optional[OidcSettings] = None
    local_source_roots: Tuple[str, ...] = ()
    allowed_ports: Tuple[int, ...] = (80, 443)
    allow_private_origins: Tuple[str, ...] = ()
    allow_private_cidrs: Tuple[str, ...] = ()
    confirm_non_loopback: bool = False

    @property
    def is_strict(self) -> bool:
        return self.mode == MODE_AUTHENTICATED_SERVER

    @property
    def auth_required(self) -> bool:
        return self.mode == MODE_AUTHENTICATED_SERVER


# ── Parsing helpers ─────────────────────────────────────────────────────────


def _parse_ports(values: Optional[List[str]], source: str) -> Tuple[int, ...]:
    if not values:
        return (80, 443)
    ports: List[int] = []
    for raw in values:
        try:
            port = int(str(raw).strip())
        except ValueError:
            raise ConfigurationError(f"{source} contains a non-integer port: {raw!r}")
        if not 1 <= port <= 65535:
            raise ConfigurationError(f"{source} ports must be between 1 and 65535.")
        if port not in ports:
            ports.append(port)
    return tuple(ports)


def _validate_cidrs(values: List[str], source: str) -> Tuple[str, ...]:
    validated: List[str] = []
    for raw in values:
        try:
            network = ipaddress.ip_network(str(raw).strip(), strict=False)
        except ValueError as exc:
            raise ConfigurationError(f"{source} contains an invalid CIDR: {raw!r}") from exc
        validated.append(str(network))
    return tuple(validated)


def _tokens_from_yaml(auth_section: Dict[str, Any]) -> List[TokenEntry]:
    entries: List[TokenEntry] = []
    for index, raw_entry in enumerate(auth_section.get("tokens", []) or []):
        if not isinstance(raw_entry, dict):
            raise ConfigurationError("server.auth.tokens entries must be mappings.")
        entry_id = str(raw_entry.get("id") or f"token-{index + 1}")
        token_value = raw_entry.get("token")
        token_hash_value = raw_entry.get("token_hash")
        if bool(token_value) == bool(token_hash_value):
            raise ConfigurationError(
                f"Token '{entry_id}' must set exactly one of 'token' (plaintext, "
                "hashed at startup) or 'token_hash' (sha256:<hex>)."
            )
        token_hash = normalize_stored_token(token_value or token_hash_value)
        scopes = parse_scopes(raw_entry.get("scopes")) or DEFAULT_TOKEN_SCOPES
        providers_raw = raw_entry.get("providers")
        providers: Optional[FrozenSet[str]] = None
        if providers_raw is not None:
            providers = frozenset(str(item) for item in providers_raw if str(item))
        entries.append(
            TokenEntry(
                id=entry_id,
                token_hash=token_hash,
                scopes=scopes,
                providers=providers,
            )
        )
    return entries


def _oidc_from_yaml(auth_section: Dict[str, Any]) -> Optional[OidcSettings]:
    raw = auth_section.get("oidc") if isinstance(auth_section, dict) else None
    if not raw:
        return None
    issuer = str(raw.get("issuer", "")).strip()
    audience = str(raw.get("audience", "")).strip()
    if not issuer or not audience:
        raise ConfigurationError(
            "server.auth.oidc requires both 'issuer' and 'audience'."
        )
    algorithms = tuple(
        str(item).strip().upper()
        for item in (raw.get("algorithms") or ["RS256", "ES256"])
        if str(item).strip()
    )
    return OidcSettings(
        issuer=issuer.rstrip("/"),
        audience=audience,
        jwks_uri=(str(raw["jwks_uri"]).strip() if raw.get("jwks_uri") else None),
        algorithms=algorithms,
        scopes_claim=str(raw.get("scopes_claim") or "scope").strip(),
        providers_claim=(
            str(raw["providers_claim"]).strip()
            if raw.get("providers_claim")
            else None
        ),
    )


def _cors_from_env_or_yaml(server_section: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    env_value = os.environ.get("SUMMARIZER_CORS_ORIGINS")
    if env_value is not None:
        if env_value.strip() == "*":
            return ("*",)
        return tuple(item.strip() for item in env_value.split(",") if item.strip()) or None
    raw = server_section.get("cors_origins") or server_section.get("cors-origins")
    if not raw:
        return None
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _validate_settings(settings: ServerSettings) -> None:
    if settings.mode not in VALID_MODES:
        raise ConfigurationError(
            f"Unknown deployment mode: {settings.mode!r}. "
            f"Valid modes: {', '.join(VALID_MODES)}"
        )
    if settings.mode != MODE_AUTHENTICATED_SERVER:
        return
    if not settings.tokens and settings.oidc is None:
        raise ConfigurationError(
            "authenticated-server mode requires authentication. Configure at "
            "least one static API token via SUMMARIZER_API_TOKEN or "
            "server.auth.tokens in summarizer.yaml, or enable OIDC via "
            "SUMMARIZER_OIDC_ISSUER + SUMMARIZER_OIDC_AUDIENCE "
            "(server.auth.oidc). To keep the legacy unauthenticated local "
            "server, set SUMMARIZER_DEPLOY_MODE=local-trusted."
        )
    if settings.oidc is not None and not settings.oidc.issuer:
        raise ConfigurationError("OIDC configuration requires an issuer.")


def _build_env_token() -> Optional[TokenEntry]:
    raw_token = os.environ.get("SUMMARIZER_API_TOKEN")
    if not raw_token or not raw_token.strip():
        return None
    scopes_value = os.environ.get("SUMMARIZER_API_TOKEN_SCOPES")
    scopes = parse_scopes(scopes_value) if scopes_value else DEFAULT_TOKEN_SCOPES
    return TokenEntry(
        id="env-token",
        token_hash=hash_token(raw_token.strip()),
        scopes=scopes,
    )


def _build_env_oidc() -> Optional[OidcSettings]:
    issuer = os.environ.get("SUMMARIZER_OIDC_ISSUER", "").strip()
    audience = os.environ.get("SUMMARIZER_OIDC_AUDIENCE", "").strip()
    if not issuer and not audience:
        return None
    if not issuer or not audience:
        raise ConfigurationError(
            "SUMMARIZER_OIDC_ISSUER and SUMMARIZER_OIDC_AUDIENCE must be set together."
        )
    algorithms_raw = os.environ.get("SUMMARIZER_OIDC_ALGORITHMS")
    algorithms = (
        tuple(item.strip().upper() for item in algorithms_raw.split(",") if item.strip())
        if algorithms_raw
        else ("RS256", "ES256")
    )
    return OidcSettings(
        issuer=issuer.rstrip("/"),
        audience=audience,
        jwks_uri=os.environ.get("SUMMARIZER_OIDC_JWKS_URI") or None,
        algorithms=algorithms,
        scopes_claim=os.environ.get("SUMMARIZER_OIDC_SCOPES_CLAIM") or "scope",
        providers_claim=os.environ.get("SUMMARIZER_OIDC_PROVIDERS_CLAIM") or None,
    )


def load_server_settings(
    mode_arg: Optional[str] = None,
    local_source_roots_arg: Optional[List[str]] = None,
    file_config: Optional[Dict[str, Any]] = None,
) -> ServerSettings:
    """Resolve the effective server settings from CLI/env/YAML/defaults.

    Args:
        mode_arg: Explicit mode from the ``serve --mode`` CLI flag.
        local_source_roots_arg: Roots from repeatable ``--local-source-root``.
        file_config: Optional pre-loaded YAML config (injected by tests).
    """
    if file_config is None:
        file_config = load_config_file() or {}
    server_section = file_config.get("server") or {}
    outbound_section = file_config.get("outbound") or {}
    if not isinstance(server_section, dict) or not isinstance(outbound_section, dict):
        raise ConfigurationError("'server' and 'outbound' YAML sections must be mappings.")

    # Mode: CLI > env > YAML > default.
    yaml_mode = server_section.get("mode")
    env_mode = os.environ.get("SUMMARIZER_DEPLOY_MODE")
    mode = mode_arg or env_mode or yaml_mode or MODE_LOCAL_TRUSTED
    mode = str(mode).strip()
    if mode not in VALID_MODES:
        raise ConfigurationError(
            f"Unknown deployment mode: {mode!r}. Valid modes: {', '.join(VALID_MODES)}"
        )
    # Compatibility is implicit for the CLI; the server factory may only run a
    # server mode, so normalize accidental misuse to local-trusted only when the
    # caller explicitly asked for compatibility (it is handled by the CLI path).
    if mode == MODE_COMPATIBILITY:
        mode = MODE_LOCAL_TRUSTED

    # Credentials: env tokens/OIDC extend YAML-configured ones.
    tokens = _tokens_from_yaml(server_section.get("auth") or {})
    env_token = _build_env_token()
    if env_token is not None:
        tokens.append(env_token)
    oidc = _build_env_oidc() or _oidc_from_yaml(server_section.get("auth") or {})

    # Local source roots: CLI > env > YAML.
    roots: List[str] = []
    if local_source_roots_arg:
        roots = [str(item) for item in local_source_roots_arg if str(item)]
    else:
        env_roots = _env_list("SUMMARIZER_LOCAL_SOURCE_ROOTS", os.pathsep)
        if env_roots:
            roots = env_roots
        else:
            raw_roots = server_section.get("local_source_roots") or server_section.get(
                "local-source-roots"
            )
            if raw_roots:
                if isinstance(raw_roots, str):
                    raw_roots = [raw_roots]
                roots = [str(item) for item in raw_roots if str(item)]

    # Outbound section: env > YAML > defaults.
    env_origins = _env_list("SUMMARIZER_OUTBOUND_ALLOW_PRIVATE_ORIGINS")
    if env_origins is not None:
        private_origins = env_origins
    else:
        private_origins = [
            str(item) for item in (outbound_section.get("allow_private_origins") or [])
        ]
    env_cidrs = _env_list("SUMMARIZER_OUTBOUND_ALLOW_PRIVATE_CIDRS")
    if env_cidrs is not None:
        private_cidrs = env_cidrs
    else:
        private_cidrs = [
            str(item) for item in (outbound_section.get("allow_private_cidrs") or [])
        ]
    env_ports = _env_list("SUMMARIZER_OUTBOUND_ALLOWED_PORTS")
    ports_source = "SUMMARIZER_OUTBOUND_ALLOWED_PORTS"
    if env_ports is None:
        env_ports = [
            str(item) for item in (outbound_section.get("allowed_ports") or [])
        ]
        ports_source = "outbound.allowed_ports"

    settings = ServerSettings(
        mode=mode,
        cors_origins=_cors_from_env_or_yaml(server_section),
        tokens=tuple(tokens),
        oidc=oidc,
        local_source_roots=tuple(roots),
        allowed_ports=_parse_ports(env_ports, ports_source),
        allow_private_origins=tuple(private_origins),
        allow_private_cidrs=_validate_cidrs(private_cidrs, "outbound.allow_private_cidrs"),
        confirm_non_loopback=(
            _env_truthy("SUMMARIZER_CONFIRM_NON_LOOPBACK_BIND")
            or bool(server_section.get("confirm_non_loopback", False))
        ),
    )
    _validate_settings(settings)
    return settings


def is_loopback_host(host: str) -> bool:
    """Return True for loopback bind addresses (best effort, no DNS lookup)."""
    host = (host or "").strip().lower()
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind_host(host: str, settings: ServerSettings) -> None:
    """Reject non-loopback binds for local-trusted mode unless explicitly confirmed.

    Authenticated-server may bind anywhere; compatibility is never passed to the
    server factory (load_server_settings normalizes it to local-trusted).
    """
    if settings.mode == MODE_LOCAL_TRUSTED and not is_loopback_host(host):
        if not settings.confirm_non_loopback:
            raise ConfigurationError(
                f"Refusing to start local-trusted mode on non-loopback host {host!r}: "
                "this mode has no authentication. Either bind a loopback address "
                "(default 127.0.0.1), pass --confirm-non-loopback-bind "
                "(SUMMARIZER_CONFIRM_NON_LOOPBACK_BIND=true), or run "
                "--mode authenticated-server with API token/OIDC configured."
            )
