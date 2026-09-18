"""Security primitives for deployment modes, authentication and outbound policy.

This package is imported lazily where possible so that the standalone CLI keeps
working with a minimal dependency set (python-jose/FastAPI are only required by
the authenticated-server mode).
"""

from .httpguards import (
    GuardedAiohttpSession,
    aiohttp_session_for,
    guarded_aiohttp_session,
    guarded_request,
    preflight_url,
    session_for,
)
from .netpolicy import (
    Origin,
    OutboundPolicy,
    OutboundPolicyError,
    PURPOSE_COBALT_API,
    PURPOSE_COBALT_DOWNLOAD,
    PURPOSE_DRIVE,
    PURPOSE_DROPBOX,
    PURPOSE_GENERIC_DOWNLOAD,
    PURPOSE_MODEL,
    PURPOSE_OIDC,
    PURPOSE_VISION,
    PURPOSE_YOUTUBE_CAPTIONS,
    build_outbound_policy,
    classify_ip,
    get_policy,
    host_matches_suffix,
    normalize_origin,
    set_policy,
)
from .settings import (
    MODE_AUTHENTICATED_SERVER,
    MODE_COMPATIBILITY,
    MODE_LOCAL_TRUSTED,
    VALID_MODES,
    OidcSettings,
    ServerSettings,
    TokenEntry,
    hash_token,
    load_server_settings,
    validate_bind_host,
)

__all__ = [
    "MODE_AUTHENTICATED_SERVER",
    "MODE_COMPATIBILITY",
    "MODE_LOCAL_TRUSTED",
    "VALID_MODES",
    "GuardedAiohttpSession",
    "OidcSettings",
    "Origin",
    "aiohttp_session_for",
    "OutboundPolicy",
    "OutboundPolicyError",
    "PURPOSE_COBALT_API",
    "PURPOSE_COBALT_DOWNLOAD",
    "PURPOSE_DRIVE",
    "PURPOSE_DROPBOX",
    "PURPOSE_GENERIC_DOWNLOAD",
    "PURPOSE_MODEL",
    "PURPOSE_OIDC",
    "PURPOSE_VISION",
    "PURPOSE_YOUTUBE_CAPTIONS",
    "ServerSettings",
    "TokenEntry",
    "build_outbound_policy",
    "classify_ip",
    "get_policy",
    "guarded_aiohttp_session",
    "guarded_request",
    "hash_token",
    "host_matches_suffix",
    "load_server_settings",
    "normalize_origin",
    "preflight_url",
    "session_for",
    "set_policy",
    "validate_bind_host",
]
