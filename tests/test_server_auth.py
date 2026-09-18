"""TR-6.x / TR-8.x tests for authentication and the hardened FastAPI app."""

import logging
import time
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt
from jose.constants import ALGORITHMS
from jose import jwk as jose_jwk

from summarizer.security.auth import (
    AuthGuard,
    OidcVerifier,
    authenticate_token,
    identity_from_claims,
    looks_like_jwt,
    make_auth_guard,
)
from summarizer.security.socketguard import (
    socket_guard_active,
    uninstall_socket_guard,
)
from summarizer.security.settings import (
    ALL_KNOWN_SCOPES,
    MODE_AUTHENTICATED_SERVER,
    MODE_LOCAL_TRUSTED,
    OidcSettings,
    SCOPE_READ,
    SCOPE_SOURCES_LOCAL,
    SCOPE_UPLOAD,
    SCOPE_WRITE,
    ServerSettings,
    TokenEntry,
    hash_token,
)
from summarizer.server import create_app

ISSUER = "https://issuer.example.test"
AUDIENCE = "summarizer-api"
KID = "test-key-1"
SECRET = "super-secret-token-value-do-not-leak"
SERVER_API_KEY = "sk-server-only-secret"

FILE_CONFIG = {
    "default_provider": "groq",
    "providers": {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
            "api_key": SERVER_API_KEY,
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash-lite",
        },
    },
    "defaults": {"cobalt-base-url": "http://localhost:9000"},
}


# ── helpers ─────────────────────────────────────────────────────────────────


def _pem_and_jwks():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    rsa_key = jose_jwk.construct(public_pem, ALGORITHMS.RS256)
    jwk_dict = rsa_key.to_dict()
    jwk_dict.update({"kid": KID, "alg": "RS256", "use": "sig"})
    return pem, {"keys": [jwk_dict]}


def _other_pem():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("utf-8")


def _make_jwt(pem, claims, kid=KID, alg="RS256"):
    return jose_jwt.encode(
        claims, pem, algorithm=alg, headers={"kid": kid, "alg": alg}
    )


def _base_claims(**overrides):
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-42",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "scope": f"{SCOPE_READ} {SCOPE_WRITE}",
    }
    claims.update(overrides)
    return claims


def _token_settings():
    return ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(
            TokenEntry(
                id="main-token",
                token_hash=hash_token(SECRET),
                scopes=frozenset({SCOPE_READ, SCOPE_WRITE}),
            ),
        ),
    )


def _oidc_settings(**overrides):
    params = {"issuer": ISSUER, "audience": AUDIENCE, **overrides}
    return OidcSettings(**params)


def _oidc_settings_with_jwks(jwks, **overrides):
    settings = _oidc_settings(**overrides)
    return settings, OidcVerifier(settings, jwks_fetcher=lambda: jwks)


def _request(authorization=None):
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    return SimpleNamespace(headers=headers)


# ── static tokens ───────────────────────────────────────────────────────────


def test_tr6_1_missing_or_malformed_headers_return_401():
    guard = make_auth_guard(_token_settings())
    for header in (None, "Basic abc", "Bearer", "Bearer  ", "Token abc"):
        with pytest.raises(HTTPException) as exc_info:
            guard.identity(_request(header))
        assert exc_info.value.status_code == 401
        assert exc_info.value.headers.get("WWW-Authenticate") == "Bearer"


def test_tr6_1_wrong_token_401_and_response_never_echoes_secret():
    submitted = "definitely-wrong-token"
    guard = make_auth_guard(_token_settings())
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {submitted}"))
    assert exc_info.value.status_code == 401
    body = str(exc_info.value.detail)
    assert submitted not in body
    assert SECRET not in body


def test_tr6_2_correct_token_yields_identity():
    guard = make_auth_guard(_token_settings())
    identity = guard.identity(_request(f"Bearer {SECRET}"))
    assert identity.method == "token"
    assert identity.subject == "main-token"
    assert SCOPE_READ in identity.scopes
    assert identity.provider_allowed("anything")  # no provider restriction


def test_token_provider_restriction():
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(
            TokenEntry(
                id="scoped-token",
                token_hash=hash_token(SECRET),
                scopes=frozenset({SCOPE_READ}),
                providers=frozenset({"gemini"}),
            ),
        ),
    )
    identity = authenticate_token(SECRET, settings)
    assert identity.provider_allowed("gemini")
    assert not identity.provider_allowed("openai")
    assert not identity.provider_allowed(None)


def test_jwt_shaped_static_token_is_accepted_as_static():
    static = "aaa.bbb.ccc"
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        oidc=_oidc_settings(),
        tokens=(
            TokenEntry(
                id="jwt-like",
                token_hash=hash_token(static),
                scopes=frozenset({SCOPE_READ}),
            ),
        ),
    )
    guard = make_auth_guard(settings)
    identity = guard.identity(_request(f"Bearer {static}"))
    assert identity.method == "token"
    assert identity.subject == "jwt-like"


# ── OIDC ────────────────────────────────────────────────────────────────────


def test_tr6_3_valid_jwt_identity():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(pem, _base_claims())
    identity = guard.identity(_request(f"Bearer {token}"))
    assert identity.method == "oidc"
    assert identity.subject == "user-42"
    assert identity.scopes == frozenset({SCOPE_READ, SCOPE_WRITE})


def test_tr6_3_bad_signature_rejected():
    pem, jwks = _pem_and_jwks()
    attacker_pem = _other_pem()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(attacker_pem, _base_claims())
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {token}"))
    assert exc_info.value.status_code == 401


def test_tr6_3_expired_jwt_rejected():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    old = _base_claims(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200)
    token = _make_jwt(pem, old)
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {token}"))
    assert exc_info.value.status_code == 401


def test_tr6_3_wrong_audience_rejected():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(pem, _base_claims(aud="some-other-api"))
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {token}"))
    assert exc_info.value.status_code == 401


def test_tr6_3_wrong_issuer_rejected():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(pem, _base_claims(iss="https://evil.example"))
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {token}"))
    assert exc_info.value.status_code == 401


def _assert_401_for_claims(mutate):
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    claims = _base_claims()
    mutate(claims)
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {_make_jwt(pem, claims)}"))
    assert exc_info.value.status_code == 401


def test_jwt_without_exp_rejected():
    _assert_401_for_claims(lambda claims: claims.pop("exp"))


def test_jwt_without_aud_rejected():
    _assert_401_for_claims(lambda claims: claims.pop("aud"))


def test_jwt_without_iat_rejected():
    _assert_401_for_claims(lambda claims: claims.pop("iat"))


def test_unknown_scope_in_claim_is_ignored():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(
        pem, _base_claims(scope=f"{SCOPE_READ} bogus:scope other:thing")
    )
    identity = guard.identity(_request(f"Bearer {token}"))
    assert identity.scopes == frozenset({SCOPE_READ})


def test_tr6_3_missing_scope_denied_403():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    read_only_claims = _base_claims(scope=SCOPE_READ)
    token = _make_jwt(pem, read_only_claims)
    identity = guard.identity(_request(f"Bearer {token}"))
    write_dependency = guard.require_scope(SCOPE_WRITE)
    with pytest.raises(HTTPException) as exc_info:
        write_dependency(identity=identity)
    assert exc_info.value.status_code == 403
    read_dependency = guard.require_scope(SCOPE_READ)
    assert read_dependency(identity=identity) is identity


def test_hs256_algorithm_confusion_rejected():
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    shared_secret = AUDIENCE  # classic public-value confusion attempt
    token = _make_jwt(shared_secret, _base_claims(), kid=KID, alg="HS256")
    with pytest.raises(HTTPException) as exc_info:
        guard.identity(_request(f"Bearer {token}"))
    assert exc_info.value.status_code == 401


def test_scope_claim_supports_list_and_providers_claim():
    pem, jwks = _pem_and_jwks()
    oidc = _oidc_settings(scopes_claim="scp", providers_claim="allowed_providers")
    verifier = OidcVerifier(oidc, jwks_fetcher=lambda: jwks)
    claims = _base_claims()
    claims["scp"] = [SCOPE_READ]
    claims["allowed_providers"] = ["gemini", "openai"]
    token = _make_jwt(pem, claims)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    identity = make_auth_guard(settings, verifier).identity(_request(f"Bearer {token}"))
    assert identity.scopes == frozenset({SCOPE_READ})
    assert identity.providers == frozenset({"gemini", "openai"})


def test_identity_from_claims_string_providers():
    oidc = _oidc_settings(providers_claim="providers")
    identity = identity_from_claims(
        {"sub": "x", "scope": SCOPE_READ, "providers": "gemini openai"}, oidc
    )
    assert identity.providers == frozenset({"gemini", "openai"})


def test_looks_like_jwt():
    assert looks_like_jwt("aaa.bbb.ccc")
    assert not looks_like_jwt("plain-token")
    assert not looks_like_jwt("aaa.bbb")


# ── log safety ──────────────────────────────────────────────────────────────


def test_tr6_4_failure_logs_never_contain_token(caplog):
    guard = make_auth_guard(_token_settings())
    tampered = f"{SECRET}-tampered-suffix"
    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException):
            guard.identity(_request(f"Bearer {tampered}"))
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert tampered not in rendered
    assert SECRET not in rendered


def test_tr6_4_oidc_failure_logs_never_contain_jwt(caplog):
    pem, jwks = _pem_and_jwks()
    oidc, verifier = _oidc_settings_with_jwks(jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    guard = make_auth_guard(settings, verifier)
    token = _make_jwt(_other_pem(), _base_claims())
    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException):
            guard.identity(_request(f"Bearer {token}"))
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert token not in rendered


# ─────────────────────────────────────────────────────────────────────────────
# TR-8.x application-level tests (authenticated-server wiring)
# ─────────────────────────────────────────────────────────────────────────────


def _strict_settings(scopes=None, providers=None, roots=()):
    return ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(
            TokenEntry(
                id="t1",
                token_hash=hash_token(SECRET),
                scopes=scopes if scopes is not None else frozenset(ALL_KNOWN_SCOPES),
                providers=providers,
            ),
        ),
        local_source_roots=tuple(str(item) for item in roots),
    )


def _strict_client(settings=None, verifier=None):
    application = create_app(
        settings or _strict_settings(),
        file_config=FILE_CONFIG,
        oidc_verifier=verifier,
    )
    return TestClient(application)


def _auth(token=SECRET):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _balance_socket_guard():
    """Strict app factories install the socket guard; TestClient does not run
    the lifespan shutdown, so release each factory install after every test."""
    yield
    while socket_guard_active():
        uninstall_socket_guard()


@pytest.fixture
def patched_config(monkeypatch):
    """Endpoints reload config per request; serve them the registered set."""
    monkeypatch.setattr(
        "summarizer.server.load_config_file", lambda: dict(FILE_CONFIG)
    )


# ── TR-8.1: endpoint × permission matrix ────────────────────────────────────


def test_tr8_1_health_anonymous_with_mode():
    client = _strict_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == MODE_AUTHENTICATED_SERVER


def test_tr8_1_docs_and_openapi_disabled_in_strict():
    client = _strict_client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_tr8_1_empty_string_override_is_rejected(patched_config):
    client = _strict_client()
    response = client.post(
        "/summarize",
        json={"source": "https://example.test/v", "base_url": ""},
        headers=_auth(),
    )
    assert response.status_code == 403
    assert "base_url" in response.text


def test_tr8_1_config_never_exposes_static_tokens(monkeypatch):
    from summarizer.server import _redact_server_secrets

    plaintext = "super-secret-token-value"
    cfg = dict(FILE_CONFIG)
    cfg["server"] = {
        "auth": {
            "tokens": [
                {"id": "t1", "token": plaintext, "scopes": ["summarize:read"]},
                {"id": "t2", "token_hash": "sha256:abcdef", "scopes": []},
            ]
        }
    }
    redacted = _redact_server_secrets(cfg)
    rendered = str(redacted)
    assert plaintext not in rendered
    assert "sha256:abcdef" not in rendered
    assert redacted["server"]["auth"]["tokens"][0]["id"] == "t1"

    monkeypatch.setattr("summarizer.server.load_config_file", lambda: cfg)
    client = _strict_client()
    response = client.get("/config", headers=_auth())
    assert response.status_code == 200
    assert plaintext not in response.text


def test_tr8_1_policy_error_in_wrapper_chain_is_sanitized():
    from summarizer.exceptions import SummarizerError
    from summarizer.security.netpolicy import OutboundPolicyError
    from summarizer.server import _client_safe_error_text

    inner = OutboundPolicyError(
        "Refusing connection to 10.10.20.30: address category 'private' "
        "is blocked by the outbound policy."
    )
    wrapped = SummarizerError("Failed to download Google Drive file")
    wrapped.__cause__ = inner
    safe = _client_safe_error_text(wrapped)
    assert "10.10.20.30" not in safe
    assert "private" not in safe
    assert "outbound security policy" in safe

    plain = SummarizerError("ordinary functional failure")
    assert _client_safe_error_text(plain) == "ordinary functional failure"


def test_tr8_1_protected_endpoints_require_credentials():
    client = _strict_client()
    for path in ("/providers", "/prompts", "/config"):
        response = client.get(path)
        assert response.status_code == 401, path
        assert response.headers.get("WWW-Authenticate") == "Bearer"
    response = client.post("/summarize", json={"source": "x"})
    assert response.status_code == 401
    assert SECRET not in response.text


def test_tr8_1_scope_matrix():
    client = _strict_client(
        _strict_settings(scopes=frozenset({SCOPE_READ}))
    )
    assert client.get("/providers", headers=_auth()).status_code == 200
    assert client.get("/config", headers=_auth()).status_code == 403
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={"source": "https://youtube.com/watch?v=x", "provider": "groq"},
    )
    assert response.status_code == 403
    assert "summarize:write" in response.text


def test_tr8_1_upload_requires_upload_scope():
    client = _strict_client(
        _strict_settings(scopes=frozenset({SCOPE_WRITE}))
    )
    response = client.post(
        "/summarize/upload",
        headers=_auth(),
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"provider": "groq"},
    )
    assert response.status_code == 403
    assert SCOPE_UPLOAD in response.text


# ── TR-8.2: provider whitelist ──────────────────────────────────────────────


def test_tr8_2_unregistered_provider_rejected(patched_config):
    client = _strict_client()
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": "https://youtube.com/watch?v=x",
            "provider": "totally-not-registered",
        },
    )
    assert response.status_code == 403


def test_tr8_2_identity_provider_restriction_enforced(patched_config):
    client = _strict_client(
        _strict_settings(providers=frozenset({"gemini"}))
    )
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": "https://youtube.com/watch?v=x",
            "provider": "groq",
        },
    )
    assert response.status_code == 403
    # Catalog is filtered too: only gemini is visible.
    listing = client.get("/providers", headers=_auth()).json()
    assert {item["name"] for item in listing} == {"gemini"}


def test_tr8_2_registered_provider_config_reaches_main(
    patched_config, monkeypatch
):
    captured = {}

    def fake_main(config):
        captured.update(config)
        return "summary text"

    monkeypatch.setattr("summarizer.server.main", fake_main)
    client = _strict_client()
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": "https://youtube.com/watch?v=x",
            "type": "YouTube Video",
            "provider": "groq",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["model"] == "llama-3.3-70b-versatile"
    # Server-side key comes from registered config, never from the request.
    assert captured.get("api_key") == SERVER_API_KEY


# ── TR-8.3: forbidden override fields ───────────────────────────────────────


@pytest.mark.parametrize("field", ["api_key", "base_url", "model", "cobalt_url"])
def test_tr8_3_forbidden_json_fields_403(patched_config, monkeypatch, field):
    calls = {"n": 0}

    def fake_main(config):
        calls["n"] += 1
        return "should never happen"

    monkeypatch.setattr("summarizer.server.main", fake_main)
    client = _strict_client()
    payload = {
        "source": "https://youtube.com/watch?v=x",
        "provider": "groq",
        field: "evil-value",
    }
    response = client.post("/summarize", headers=_auth(), json=payload)
    assert response.status_code == 403
    assert calls["n"] == 0


def test_tr8_3_forbidden_batch_field_403(patched_config, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        "summarizer.server.main", lambda config: calls.__setitem__("n", calls["n"] + 1)
    )
    client = _strict_client()
    response = client.post(
        "/summarize/batch",
        headers=_auth(),
        json={
            "sources": ["https://youtube.com/watch?v=x"],
            "provider": "groq",
            "model": "injected-model",
        },
    )
    assert response.status_code == 403
    assert calls["n"] == 0


@pytest.mark.parametrize("field", ["api_key", "base_url", "model", "cobalt_url"])
def test_tr8_3_forbidden_multipart_fields_403(patched_config, field):
    client = _strict_client(
        _strict_settings(
            scopes=frozenset({SCOPE_UPLOAD, SCOPE_READ, SCOPE_WRITE})
        )
    )
    response = client.post(
        "/summarize/upload",
        headers=_auth(),
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"provider": "groq", field: "evil-value"},
    )
    assert response.status_code == 403


# ── TR-8.4: local path sources ──────────────────────────────────────────────


def test_tr8_4_local_path_without_scope_403(patched_config, tmp_path):
    settings = _strict_settings(
        scopes=frozenset({SCOPE_READ, SCOPE_WRITE}), roots=(tmp_path,)
    )
    client = _strict_client(settings)
    media = tmp_path / "clip.mp4"
    media.write_text("x")
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": str(media),
            "type": "Local File",
            "provider": "groq",
        },
    )
    assert response.status_code == 403
    assert SCOPE_SOURCES_LOCAL in response.text


def test_tr8_4_local_path_outside_roots_403(patched_config, tmp_path):
    client = _strict_client(_strict_settings(roots=(tmp_path,)))
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": "/etc/passwd",
            "type": "Local File",
            "provider": "groq",
        },
    )
    assert response.status_code == 403


def test_tr8_4_local_path_inside_root_reaches_main(
    patched_config, monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(
        "summarizer.server.main", lambda config: captured.update(config) or "ok"
    )
    client = _strict_client(_strict_settings(roots=(tmp_path,)))
    media = tmp_path / "clip.mp4"
    media.write_text("x")
    response = client.post(
        "/summarize",
        headers=_auth(),
        json={
            "source": str(media),
            "type": "Local File",
            "provider": "groq",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["source_url_or_path"] == str(media.resolve())


def test_tr8_4_upload_tempfile_bypasses_roots(patched_config, monkeypatch):
    monkeypatch.setattr("summarizer.server.main", lambda config: "uploaded ok")
    client = _strict_client(
        _strict_settings(
            scopes=frozenset({SCOPE_UPLOAD, SCOPE_READ, SCOPE_WRITE}),
            roots=(),  # no local roots configured at all
        )
    )
    response = client.post(
        "/summarize/upload",
        headers=_auth(),
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"provider": "groq"},
    )
    assert response.status_code == 200, response.text


# ── TR-8.5: admin config + secret hygiene ───────────────────────────────────


def test_tr8_5_config_requires_admin_and_is_redacted(patched_config):
    client = _strict_client()
    assert client.get("/config").status_code == 401
    response = client.get("/config", headers=_auth())
    assert response.status_code == 200
    body = response.text
    assert SERVER_API_KEY not in body
    assert "REDACTED" in body
    assert SECRET not in body


def test_tr8_5_failure_responses_and_logs_hide_secrets(patched_config, caplog):
    client = _strict_client()
    with caplog.at_level(logging.WARNING):
        response = client.get(
            "/providers", headers={"Authorization": f"Bearer {SECRET}-wrong"}
        )
    assert response.status_code == 401
    assert SECRET not in response.text
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET not in rendered


# ── OIDC end-to-end through the app ─────────────────────────────────────────


def test_tr8_oidc_jwt_authenticates_at_endpoint(patched_config):
    pem, jwks = _pem_and_jwks()
    oidc = _oidc_settings()
    verifier = OidcVerifier(oidc, jwks_fetcher=lambda: jwks)
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER, oidc=oidc)
    client = _strict_client(settings, verifier=verifier)
    token = _make_jwt(pem, _base_claims())
    response = client.get(
        "/providers", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert {item["name"] for item in response.json()} == {"groq", "gemini"}


# ── TR-8.6: local-trusted keeps historical behavior ─────────────────────────


def test_tr8_6_local_trusted_anonymous_and_overrides_accepted(
    patched_config, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "summarizer.server.main", lambda config: captured.update(config) or "ok"
    )
    application = create_app(
        ServerSettings(mode=MODE_LOCAL_TRUSTED), file_config=FILE_CONFIG
    )
    client = TestClient(application)
    assert client.get("/providers").status_code == 200
    response = client.post(
        "/summarize",
        json={
            "source": "https://youtube.com/watch?v=x",
            "type": "YouTube Video",
            "base_url": "https://custom.example.com/v1",
            "model": "custom-model",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["base_url"] == "https://custom.example.com/v1"
    assert captured["model"] == "custom-model"
