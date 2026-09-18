"""FastAPI HTTP server for the summarizer package.

Exposes all CLI functionality via a REST API. Auto-generated docs at /docs.

Deployment modes (see summarizer/security/settings.py):

* ``local-trusted`` (default): no authentication, permissive outbound policy;
  historical single-user behavior.
* ``authenticated-server``: every endpoint except ``/health`` requires an API
  token or OIDC identity with the appropriate scope; providers are restricted
  to the server-registered set; request overrides of api_key/base_url/model/
  cobalt_url are forbidden; local path sources require sources:local scope and
  configured roots; the strict outbound policy is enforced.
"""

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
try:
    from pydantic import BaseModel, Field, model_validator

    def _before_model_validator(func):
        return model_validator(mode="before")(classmethod(func))

except ImportError:
    from pydantic import BaseModel, Field, root_validator

    def _before_model_validator(func):
        return root_validator(pre=True)(func)

from summarizer.config_file import load_config_file, merge_configs, find_config_file
from summarizer.core import main
from summarizer.exceptions import (
    ConfigurationError,
    LocalSourceDenied,
    SummarizerError,
)
from summarizer.prompts import get_available_prompts
from summarizer.api_utils import (
    build_runtime_config,
    format_output,
    SOURCE_TYPES,
    OUTPUT_FORMATS,
    TRANSCRIPTION_METHODS,
    WHISPER_MODELS,
    DEFAULT_MAX_UPLOAD_MB,
    redact_config_response,
)
from summarizer.security.auth import Identity, make_auth_guard
from summarizer.security.localfs import (
    authorize_local_source,
    is_local_source_type,
)
from summarizer.security.netpolicy import (
    OutboundPolicyError,
    build_outbound_policy,
    normalize_origin,
    reset_policy,
    set_policy,
)
from summarizer.security.settings import (
    SCOPE_ADMIN_CONFIG,
    SCOPE_READ,
    SCOPE_SOURCES_LOCAL,
    SCOPE_UPLOAD,
    SCOPE_WRITE,
    ServerSettings,
    load_server_settings,
)
from summarizer.security.socketguard import (
    install_socket_guard,
    uninstall_socket_guard,
)

logger = logging.getLogger(__name__)

# Request-controlled fields that only the server configuration may supply in
# authenticated-server mode. A request presenting any of these is rejected
# before reaching the summarizer core.
FORBIDDEN_OVERRIDE_FIELDS = ("api_key", "base_url", "model", "cobalt_url")


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

SourceType = Literal[tuple(SOURCE_TYPES)]  # type: ignore[misc]
OutputFormat = Literal[tuple(OUTPUT_FORMATS)]  # type: ignore[misc]
TranscriptionMethod = Literal[tuple(TRANSCRIPTION_METHODS)]  # type: ignore[misc]
WhisperModel = Literal[tuple(WHISPER_MODELS)]  # type: ignore[misc]


def _reject_legacy_audio_speed_field(data: Any) -> Any:
    if isinstance(data, dict) and "audio_speed" in data:
        raise ValueError("audio_speed is no longer supported; use speed instead")
    return data


class SummarizeRequest(BaseModel):
    source: str = Field(..., description="Video URL or file path")
    type: SourceType = Field("YouTube Video", description="Source type")
    provider: Optional[str] = Field(None, description="Provider name from config")
    prompt_type: Optional[str] = Field(None, description="Summary style")
    chunk_size: Optional[int] = Field(
        None, ge=100, le=500_000, description="Characters per chunk"
    )
    parallel_calls: Optional[int] = Field(
        None, ge=1, le=200, description="Concurrent API requests"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=1_000_000, description="Max output tokens per chunk"
    )
    language: Optional[str] = Field(None, description="Caption/transcription language")
    output_language: Optional[str] = Field(None, description="Summary output language")
    force_download: bool = Field(False, description="Skip captions, download audio")
    transcription: Optional[TranscriptionMethod] = Field(
        None, description="Cloud Whisper or Local Whisper"
    )
    whisper_model: Optional[WhisperModel] = Field(None, description="Whisper model size")
    speed: Optional[float] = Field(
        None, gt=0.0, le=10.0, description="Playback speed for audio preprocessing or visual-mode video"
    )
    output_format: OutputFormat = Field("markdown", description="markdown, json, or html")
    visual: bool = Field(False, description="Send video directly to vision model")
    use_proxy: Optional[bool] = Field(None, description="Route through the configured HTTP proxy")
    api_key: Optional[str] = Field(None, description="Override API key")
    base_url: Optional[str] = Field(None, description="Override API base URL")
    model: Optional[str] = Field(None, description="Override model name")
    cobalt_url: Optional[str] = Field(None, description="Cobalt base URL")
    verbose: bool = Field(False, description="Verbose progress output")

    @_before_model_validator
    def _reject_legacy_audio_speed(cls, data: Any) -> Any:
        return _reject_legacy_audio_speed_field(data)


class SummarizeResponse(BaseModel):
    success: bool
    source: str
    summary: str
    format: str
    model: Optional[str] = None
    prompt_type: Optional[str] = None
    processing_time_seconds: float
    error: Optional[str] = None
    error_type: Optional[str] = None


class BatchRequest(BaseModel):
    sources: List[str] = Field(..., min_length=1, description="List of URLs or file paths")
    type: SourceType = Field("YouTube Video", description="Source type for all items")
    provider: Optional[str] = Field(None, description="Provider name from config")
    prompt_type: Optional[str] = Field(None, description="Summary style")
    chunk_size: Optional[int] = Field(
        None, ge=100, le=500_000, description="Characters per chunk"
    )
    parallel_calls: Optional[int] = Field(
        None, ge=1, le=200, description="Concurrent API requests"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=1_000_000, description="Max output tokens per chunk"
    )
    language: Optional[str] = Field(None, description="Caption/transcription language")
    output_language: Optional[str] = Field(None, description="Summary output language")
    force_download: bool = Field(False, description="Skip captions, download audio")
    transcription: Optional[TranscriptionMethod] = Field(
        None, description="Cloud Whisper or Local Whisper"
    )
    whisper_model: Optional[WhisperModel] = Field(None, description="Whisper model size")
    speed: Optional[float] = Field(
        None, gt=0.0, le=10.0, description="Playback speed for audio preprocessing or visual-mode video"
    )
    output_format: OutputFormat = Field("markdown", description="markdown, json, or html")
    visual: bool = Field(False, description="Send video directly to vision model")
    use_proxy: Optional[bool] = Field(None, description="Route through the configured HTTP proxy")
    api_key: Optional[str] = Field(None, description="Override API key")
    base_url: Optional[str] = Field(None, description="Override API base URL")
    model: Optional[str] = Field(None, description="Override model name")
    cobalt_url: Optional[str] = Field(None, description="Cobalt base URL")
    verbose: bool = Field(False, description="Verbose progress output")

    @_before_model_validator
    def _reject_legacy_audio_speed(cls, data: Any) -> Any:
        return _reject_legacy_audio_speed_field(data)


class BatchResult(BaseModel):
    source: str
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    processing_time_seconds: float


class BatchResponse(BaseModel):
    success_count: int
    total_count: int
    results: List[BatchResult]
    overall_processing_time_seconds: float


class ProviderInfo(BaseModel):
    name: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    chunk_size: Optional[int] = None


class ConfigResponse(BaseModel):
    default_provider: Optional[str] = None
    providers: Dict[str, Any]
    defaults: Dict[str, Any]
    config_file_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

SNAKE_OVERRIDES = {
    "provider": "provider",
    "api_key": "api_key",
    "base_url": "base_url",
    "model": "model",
    "prompt_type": "prompt_type",
    "chunk_size": "chunk_size",
    "parallel_calls": "parallel_api_calls",
    "max_tokens": "max_output_tokens",
    "language": "language",
    "output_language": "output_language",
    "transcription": "transcription_method",
    "whisper_model": "whisper_model",
    "speed": "speed",
    "cobalt_url": "cobalt_base_url",
    "use_proxy": "use_proxy",
    "visual": "visual",
}


def _build_overrides(req: SummarizeRequest) -> Dict[str, Any]:
    """Build a CLI-args-style dict from a request for merge_configs."""
    overrides: Dict[str, Any] = {}
    for field, target in SNAKE_OVERRIDES.items():
        value = getattr(req, field)
        if value is not None:
            overrides[target] = value
    return overrides


def _build_runtime_config_from_request(
    req: SummarizeRequest,
    source_override: Optional[str] = None,
    type_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build runtime config from request, using the same merge path as CLI."""
    file_config = load_config_file()
    overrides = _build_overrides(req)
    merged = merge_configs(file_config, overrides)
    # Ensure per-request overrides win even when merge_configs is mocked in tests.
    merged.update(overrides)

    # Match the guard that exists in the CLI after merging (gives actionable errors
    # for Raycast / API users when they specify a provider name).
    if not merged.get("base_url") or not merged.get("model"):
        prov = overrides.get("provider")
        if prov:
            raise ConfigurationError(
                f"Provider '{prov}' not found in config file (or the provider section is missing base_url/model). "
                f"Check your summarizer.yaml or set base_url + model directly."
            )
        raise ConfigurationError(
            "base_url and model are required. Either use a named 'provider' that exists in summarizer.yaml, "
            "or provide base_url and model explicitly."
        )

    return build_runtime_config(
        merged=merged,
        source=source_override or req.source,
        type_of_source=type_override or req.type,
        verbose=req.verbose,
        force_download=req.force_download,
    )


def _chain_has_outbound_policy_error(exc: BaseException) -> bool:
    """True when exc (or any chained cause/context) is an OutboundPolicyError.

    Downloader layers wrap low-level network failures in SummarizerError; the
    wrapped policy message may quote internal IPs and must not reach clients.
    """
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutboundPolicyError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _client_safe_error_text(exc: Exception) -> str:
    if _chain_has_outbound_policy_error(exc):
        logger.warning("Outbound policy violation surfaced from error chain: %s", exc)
        return "The request was blocked by the outbound security policy."
    return str(exc)


def _error_response(
    source: str,
    output_format: str,
    elapsed: float,
    exc: Exception,
) -> SummarizeResponse:
    """Build a structured error response."""
    return SummarizeResponse(
        success=False,
        source=source,
        summary="",
        format=output_format,
        error=_client_safe_error_text(exc),
        error_type=exc.__class__.__name__,
        processing_time_seconds=round(elapsed, 2),
    )


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ──────────────────────────────────────────────────────────────────────────────


def _registered_provider_origins(
    file_config: Dict[str, Any],
) -> List[tuple]:
    """Return (name, Origin) for registered http(s) providers.

    The ``litellm`` sentinel base URL has no fixed HTTP origin and is covered by
    the socket-level guard instead.
    """
    origins: List[tuple] = []
    providers_cfg = file_config.get("providers") or {}
    if not isinstance(providers_cfg, dict):
        return origins
    for name, cfg in providers_cfg.items():
        if not isinstance(cfg, dict):
            continue
        raw_url = cfg.get("base_url")
        if not raw_url or raw_url == "litellm":
            continue
        try:
            origins.append((str(name), normalize_origin(str(raw_url))))
        except OutboundPolicyError as exc:
            logger.warning(
                "Registered provider %r has an invalid base URL origin; it will "
                "not be reachable through the outbound policy: %s",
                name,
                exc,
            )
    return origins


def _configured_cobalt_url(
    file_config: Dict[str, Any], *, strict: bool = False
) -> Optional[str]:
    """Explicitly configured Cobalt origin, if any.

    In strict mode a missing configuration resolves to None rather than the
    historical ``http://localhost:9000`` default: an implicit loopback
    exemption must never be attached to every authenticated deployment.
    """
    defaults = file_config.get("defaults") or {}
    value = (
        defaults.get("cobalt-base-url")
        or defaults.get("cobalt_base_url")
        or file_config.get("cobalt_base_url")
    )
    if value:
        return str(value)
    return None if strict else "http://localhost:9000"


def _configured_proxy_endpoints() -> List[Tuple[str, int]]:
    """Return (hostname, port) pairs for every configured proxy URL."""
    try:
        from summarizer.proxy import get_proxies

        proxies = get_proxies(True) or {}
    except Exception as exc:  # misconfigured proxy is fatal later; never here
        logger.warning("Could not enumerate configured proxy hosts: %s", exc)
        return []
    endpoints: List[Tuple[str, int]] = []
    for value in proxies.values():
        if not value:
            continue
        try:
            parsed = urlparse(str(value))
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            continue
        if host and port is not None:
            endpoints.append((host, port))
    return endpoints


def _build_policy(settings: ServerSettings, file_config: Dict[str, Any]):
    strict = settings.is_strict
    cobalt_origin = None
    cobalt_url = _configured_cobalt_url(file_config, strict=strict)
    if cobalt_url:
        try:
            cobalt_origin = normalize_origin(cobalt_url)
        except OutboundPolicyError as exc:
            logger.warning("Configured Cobalt URL has an invalid origin: %s", exc)
    oidc_origin = None
    oidc_extra_origins: List[Any] = []
    if settings.oidc is not None and settings.oidc.enabled:
        try:
            oidc_origin = normalize_origin(settings.oidc.issuer)
        except OutboundPolicyError as exc:
            logger.warning("Configured OIDC issuer has an invalid origin: %s", exc)
        # An explicitly configured JWKS endpoint may live on another origin
        # than the issuer (Google/Microsoft style deployments).
        if getattr(settings.oidc, "jwks_uri", ""):
            try:
                oidc_extra_origins.append(
                    normalize_origin(settings.oidc.jwks_uri)
                )
            except OutboundPolicyError as exc:
                logger.warning("Configured JWKS URI has an invalid origin: %s", exc)
    return build_outbound_policy(
        settings,
        provider_origins=_registered_provider_origins(file_config),
        cobalt_origin=cobalt_origin,
        oidc_origin=oidc_origin,
        oidc_extra_origins=oidc_extra_origins,
        proxy_endpoints=_configured_proxy_endpoints(),
    )


def _redact_server_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove static token material from the server section before /config."""
    server_section = config.get("server")
    if not isinstance(server_section, dict):
        return config
    auth_section = server_section.get("auth")
    if not isinstance(auth_section, dict):
        return config
    tokens = auth_section.get("tokens")
    if isinstance(tokens, list):
        redacted_tokens = []
        for entry in tokens:
            if not isinstance(entry, dict):
                continue
            safe_entry = dict(entry)
            if "token" in safe_entry:
                safe_entry["token"] = "***REDACTED***"
            if "token_hash" in safe_entry:
                safe_entry["token_hash"] = "***REDACTED***"
            redacted_tokens.append(safe_entry)
        auth_section["tokens"] = redacted_tokens
    return config


def _reject_forbidden_fields(req: Any) -> None:
    for field in FORBIDDEN_OVERRIDE_FIELDS:
        value = getattr(req, field, None)
        if value is not None:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Field '{field}' is server-managed in authenticated-server "
                    "mode; request overrides are forbidden. Select a registered "
                    "provider via 'provider' instead."
                ),
            )


def _enforce_registered_provider(
    req: Any, identity: Identity, file_config: Dict[str, Any]
) -> str:
    providers_cfg = file_config.get("providers") or {}
    selected = req.provider or file_config.get("default_provider")
    if not selected or selected not in providers_cfg:
        raise HTTPException(
            status_code=403,
            detail=(
                "Only server-registered provider ids may be used. Request the "
                "catalog via GET /providers."
            ),
        )
    if not identity.provider_allowed(selected):
        raise HTTPException(
            status_code=403,
            detail="The presented identity is not authorized for this provider.",
        )
    return str(selected)


def _enforce_local_sources(
    sources: List[str], source_types: List[str], identity: Identity,
    settings: ServerSettings,
) -> None:
    for source, source_type in zip(sources, source_types):
        if not is_local_source_type(source_type):
            continue
        try:
            authorize_local_source(
                source, settings=settings, identity=identity
            )
        except LocalSourceDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc


def create_app(
    settings: Optional[ServerSettings] = None,
    allow_origins: Optional[List[str]] = None,
    *,
    file_config: Optional[Dict[str, Any]] = None,
    oidc_verifier: Optional[Any] = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: Resolved ServerSettings; loaded from CLI/env/YAML when None.
        allow_origins: Explicit CORS origin allowlist (kept for backwards
            compatibility); falls back to settings then SUMMARIZER_CORS_ORIGINS.
        file_config: Pre-loaded YAML config (injected by tests).
        oidc_verifier: Optional OidcVerifier (injected by tests).
    """
    if file_config is None:
        file_config = load_config_file() or {}
    if settings is None:
        settings = load_server_settings(file_config=file_config)
    strict = settings.is_strict

    policy = _build_policy(settings, file_config)
    if strict and oidc_verifier is None and settings.oidc is not None:
        # One shared verifier so the JWKS cache survives across requests.
        from summarizer.security.auth import OidcVerifier

        oidc_verifier = OidcVerifier(settings.oidc, policy=policy)
    guard = make_auth_guard(settings, oidc_verifier) if strict else None
    if strict:
        # Reference-counted. The fallback keeps enforcement active in worker
        # threads that third-party libraries create without the request
        # ContextVar; per-request policies still take precedence.
        install_socket_guard(fallback_policy=policy)
        registered_names = sorted((file_config.get("providers") or {}).keys())
        logger.info(
            "Starting authenticated-server: %d registered provider(s) %s, "
            "%d local source root(s), %d exempt host(s).",
            len(registered_names),
            registered_names,
            len(settings.local_source_roots),
            len(policy.exempt_hosts),
        )

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if strict:
            # Release the lifespan-time guard (factory installs another for the
            # module app; the hooks remain safe no-ops without a strict policy).
            uninstall_socket_guard()

    application = FastAPI(
        title="Summarize API",
        description="Transcribe and summarize videos from any source using any OpenAI-compatible LLM.",
        version="0.1.0",
        lifespan=lifespan,
        # In authenticated-server mode the schema/docs would disclose the
        # full attack surface anonymously; only /health stays public.
        docs_url=None if strict else "/docs",
        redoc_url=None if strict else "/redoc",
        openapi_url=None if strict else "/openapi.json",
    )
    application.state.server_settings = settings
    application.state.outbound_policy = policy

    if allow_origins is None and settings.cors_origins:
        cors_value = list(settings.cors_origins)
        allow_origins = ["*"] if cors_value == ["*"] else cors_value
    if allow_origins is None:
        origins_env = os.getenv("SUMMARIZER_CORS_ORIGINS", "")
        if origins_env == "*":
            allow_origins = ["*"]
        else:
            allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

    if allow_origins:
        # Credentials cannot be used with wildcard origins.
        allow_credentials = "*" not in allow_origins
        application.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def _policy_context():
        # Async so setup and teardown share the request task context; the value
        # is copied into run_in_threadpool workers by anyio.
        token = set_policy(policy)
        try:
            yield
        finally:
            reset_policy(token)

    def _protected(*required_scopes: str):
        """Dependency: install request policy, authenticate and check scopes."""

        async def dependency(
            request: Request, _ctx: Any = Depends(_policy_context)
        ) -> Identity:
            assert guard is not None
            identity = await run_in_threadpool(guard.identity, request)
            missing = [
                scope
                for scope in required_scopes
                if not identity.has_scope(scope)
            ]
            if missing:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Missing required scope(s): {', '.join(missing)}."
                    ),
                )
            return identity

        return dependency

    read_dep = _protected(SCOPE_READ) if strict else _policy_context
    write_dep = _protected(SCOPE_WRITE) if strict else _policy_context
    upload_dep = _protected(SCOPE_UPLOAD) if strict else _policy_context
    config_dep = _protected(SCOPE_ADMIN_CONFIG) if strict else _policy_context

    @application.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint (always anonymous)."""
        return {"status": "ok", "service": "summarize", "mode": settings.mode}

    @application.get("/providers")
    async def providers(
        identity: Optional[Identity] = Depends(read_dep),
    ) -> List[ProviderInfo]:
        """List all configured providers from summarizer.yaml."""
        file_config = load_config_file()
        providers_cfg = file_config.get("providers", {})
        result = []
        for name, cfg in providers_cfg.items():
            if strict and identity is not None and not identity.provider_allowed(name):
                continue
            result.append(ProviderInfo(
                name=name,
                base_url=cfg.get("base_url"),
                model=cfg.get("model"),
                chunk_size=cfg.get("chunk_size"),
            ))
        return result

    @application.get("/prompts")
    async def prompts(
        identity: Optional[Identity] = Depends(read_dep),
    ) -> List[str]:
        """List all available summary prompt types."""
        return get_available_prompts()

    @application.get("/config")
    async def config(
        identity: Optional[Identity] = Depends(config_dep),
    ) -> ConfigResponse:
        """Get the active merged configuration (sensitive keys redacted)."""
        file_config = load_config_file()
        safe_config = _redact_server_secrets(redact_config_response(file_config))
        defaults = safe_config.get("defaults", {})
        providers_cfg = safe_config.get("providers", {})
        default_provider = safe_config.get("default_provider")
        config_path = find_config_file()
        return ConfigResponse(
            default_provider=default_provider,
            providers=providers_cfg,
            defaults=defaults,
            config_file_path=config_path.as_posix() if config_path else None,
        )

    @application.post("/summarize", response_model=SummarizeResponse)
    async def summarize(
        req: SummarizeRequest,
        identity: Optional[Identity] = Depends(write_dep),
    ) -> SummarizeResponse:
        """Summarize a video from a URL or file path."""
        start_time = time.time()

        try:
            if strict:
                request_config = load_config_file() or {}
                _reject_forbidden_fields(req)
                _enforce_registered_provider(req, identity, request_config)
                _enforce_local_sources(
                    [req.source], [req.type], identity, settings
                )
            config = _build_runtime_config_from_request(req)
            summary = await run_in_threadpool(main, config)
            formatted = format_output(
                summary,
                req.source,
                req.output_format,
                {"prompt_type": config.get("prompt_type", ""), "model": config.get("model", "")},
            )
            elapsed = time.time() - start_time

            return SummarizeResponse(
                success=True,
                source=req.source,
                summary=formatted,
                format=req.output_format,
                model=config.get("model"),
                prompt_type=config.get("prompt_type"),
                processing_time_seconds=round(elapsed, 2),
            )

        except HTTPException:
            # Authz/validation rejections must surface as their HTTP status.
            raise
        except OutboundPolicyError as exc:
            # Do not leak internal hostnames/IPs through error responses.
            logger.warning("Outbound policy violation on /summarize: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=(
                    "The request was blocked by the outbound security policy."
                ),
            ) from exc
        except SummarizerError as e:
            elapsed = time.time() - start_time
            return _error_response(req.source, req.output_format, elapsed, e)
        except Exception as e:
            elapsed = time.time() - start_time
            return _error_response(req.source, req.output_format, elapsed, e)

    @application.post("/summarize/upload", response_model=SummarizeResponse)
    async def summarize_upload(
        file: UploadFile = File(..., description="Video or text file to summarize"),
        type: Optional[str] = Form(None, description="Source type (auto-detected if omitted)"),
        provider: Optional[str] = Form(None),
        prompt_type: Optional[str] = Form(None),
        chunk_size: Optional[int] = Form(None),
        parallel_calls: Optional[int] = Form(None),
        max_tokens: Optional[int] = Form(None),
        language: Optional[str] = Form(None),
        output_language: Optional[str] = Form(None),
        force_download: bool = Form(False),
        transcription: Optional[str] = Form(None),
        whisper_model: Optional[str] = Form(None),
        speed: Optional[float] = Form(None),
        output_format: str = Form("markdown"),
        visual: bool = Form(False),
        use_proxy: Optional[bool] = Form(None),
        api_key: Optional[str] = Form(None),
        base_url: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        cobalt_url: Optional[str] = Form(None),
        verbose: bool = Form(False),
        identity: Optional[Identity] = Depends(upload_dep),
    ) -> SummarizeResponse:
        """Summarize an uploaded file.

        Accepts video files (.mp4, .mp3, .wav, .m4a, .webm) or text files
        (.txt, .md, .vtt, .srt, .csv, .log, .rst, .html, .xml, .json).
        Text files bypass audio processing entirely.
        """
        start_time = time.time()
        tmp_path: Optional[str] = None

        if strict:
            # Reject server-managed fields and unregistered providers before
            # accepting any file bytes. Uploads use a server-managed temp file
            # and therefore bypass the local source root check.
            for field_name, field_value in (
                ("api_key", api_key),
                ("base_url", base_url),
                ("model", model),
                ("cobalt_url", cobalt_url),
            ):
                if field_value is not None:
                    raise HTTPException(
                        status_code=403,
                        detail=(
                            f"Field '{field_name}' is server-managed in "
                            "authenticated-server mode; request overrides are "
                            "forbidden. Select a registered provider via "
                            "'provider' instead."
                        ),
                    )
            upload_probe = SummarizeRequest(
                source="upload", provider=provider
            )
            _enforce_registered_provider(
                upload_probe, identity, load_config_file() or {}
            )

        # Determine source type from extension if not provided
        filename = file.filename or "upload"
        ext = Path(filename).suffix.lower()
        text_extensions = {
            ".txt", ".md", ".vtt", ".srt", ".csv",
            ".log", ".rst", ".html", ".xml", ".json",
        }
        detected_type = type or ("TXT" if ext in text_extensions else "Local File")

        try:
            # Stream upload to temp file in chunks to avoid loading large files into memory
            max_upload_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
            suffix = ext or ".bin"
            total_read = 0
            stream_chunk_size = 1024 * 1024  # 1 MB chunks

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
                while True:
                    chunk = await file.read(stream_chunk_size)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    if total_read > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum upload size of {DEFAULT_MAX_UPLOAD_MB} MB",
                        )
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Build request and config
            req = SummarizeRequest(
                source=tmp_path,
                type=detected_type,  # type: ignore[arg-type]
                provider=provider,
                prompt_type=prompt_type,
                chunk_size=chunk_size,
                parallel_calls=parallel_calls,
                max_tokens=max_tokens,
                language=language,
                output_language=output_language,
                force_download=force_download,
                transcription=transcription,  # type: ignore[arg-type]
                whisper_model=whisper_model,  # type: ignore[arg-type]
                speed=speed,
                output_format=output_format,  # type: ignore[arg-type]
                visual=visual,
                use_proxy=use_proxy,
                api_key=api_key,
                base_url=base_url,
                model=model,
                cobalt_url=cobalt_url,
                verbose=verbose,
            )
            config = _build_runtime_config_from_request(req, source_override=tmp_path, type_override=detected_type)
            summary = await run_in_threadpool(main, config)
            formatted = format_output(
                summary,
                filename,
                output_format,
                {"prompt_type": config.get("prompt_type", ""), "model": config.get("model", "")},
            )
            elapsed = time.time() - start_time

            return SummarizeResponse(
                success=True,
                source=filename,
                summary=formatted,
                format=output_format,
                model=config.get("model"),
                prompt_type=config.get("prompt_type"),
                processing_time_seconds=round(elapsed, 2),
            )

        except HTTPException:
            raise
        except OutboundPolicyError as exc:
            logger.warning("Outbound policy violation on /upload: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=(
                    "The request was blocked by the outbound security policy."
                ),
            ) from exc
        except SummarizerError as e:
            elapsed = time.time() - start_time
            return _error_response(filename, output_format, elapsed, e)
        except Exception as e:
            elapsed = time.time() - start_time
            return _error_response(filename, output_format, elapsed, e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @application.post("/summarize/batch", response_model=BatchResponse)
    async def summarize_batch(
        req: BatchRequest,
        identity: Optional[Identity] = Depends(write_dep),
    ) -> BatchResponse:
        """Summarize multiple sources in one request.

        Each source is processed sequentially. Results are returned in the same
        order as the input sources list.
        """
        if strict:
            # Validate the whole batch up front so a forbidden field or an
            # unauthorized local path never triggers partial processing.
            request_config = load_config_file() or {}
            _reject_forbidden_fields(req)
            _enforce_registered_provider(req, identity, request_config)
            _enforce_local_sources(
                list(req.sources),
                [req.type] * len(req.sources),
                identity,
                settings,
            )

        overall_start = time.time()
        results: List[BatchResult] = []

        for source in req.sources:
            item_start = time.time()
            try:
                single_req = SummarizeRequest(
                    source=source,
                    type=req.type,
                    provider=req.provider,
                    prompt_type=req.prompt_type,
                    chunk_size=req.chunk_size,
                    parallel_calls=req.parallel_calls,
                    max_tokens=req.max_tokens,
                    language=req.language,
                    output_language=req.output_language,
                    force_download=req.force_download,
                    transcription=req.transcription,
                    whisper_model=req.whisper_model,
                    speed=req.speed,
                    output_format=req.output_format,
                    visual=req.visual,
                    use_proxy=req.use_proxy,
                    api_key=req.api_key,
                    base_url=req.base_url,
                    model=req.model,
                    cobalt_url=req.cobalt_url,
                    verbose=req.verbose,
                )
                config = _build_runtime_config_from_request(single_req)
                summary = await run_in_threadpool(main, config)
                formatted = format_output(
                    summary,
                    source,
                    req.output_format,
                    {"prompt_type": config.get("prompt_type", ""), "model": config.get("model", "")},
                )
                elapsed = time.time() - item_start
                results.append(BatchResult(
                    source=source,
                    success=True,
                    summary=formatted,
                    processing_time_seconds=round(elapsed, 2),
                ))
            except OutboundPolicyError as exc:
                elapsed = time.time() - item_start
                logger.warning(
                    "Outbound policy violation in batch item: %s", exc
                )
                results.append(BatchResult(
                    source=source,
                    success=False,
                    error=(
                        "Blocked by the outbound security policy."
                    ),
                    error_type=exc.__class__.__name__,
                    processing_time_seconds=round(elapsed, 2),
                ))
            except SummarizerError as e:
                elapsed = time.time() - item_start
                results.append(BatchResult(
                    source=source,
                    success=False,
                    error=_client_safe_error_text(e),
                    error_type=e.__class__.__name__,
                    processing_time_seconds=round(elapsed, 2),
                ))
            except Exception as e:
                elapsed = time.time() - item_start
                detail = _client_safe_error_text(e)
                results.append(BatchResult(
                    source=source,
                    success=False,
                    error=f"Unexpected error: {detail}",
                    error_type=e.__class__.__name__,
                    processing_time_seconds=round(elapsed, 2),
                ))

        success_count = sum(1 for r in results if r.success)
        overall_elapsed = time.time() - overall_start

        return BatchResponse(
            success_count=success_count,
            total_count=len(req.sources),
            results=results,
            overall_processing_time_seconds=round(overall_elapsed, 2),
        )

    return application


# Default app instance used by uvicorn and production imports.
app = create_app()
