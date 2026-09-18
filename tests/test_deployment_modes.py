"""Tests for deployment-mode resolution and server security settings."""

import pytest

from summarizer.exceptions import ConfigurationError
from summarizer.security import (
    MODE_AUTHENTICATED_SERVER,
    MODE_LOCAL_TRUSTED,
    hash_token,
    load_server_settings,
    validate_bind_host,
)
from summarizer.security.settings import (
    ALL_KNOWN_SCOPES,
    SCOPE_WRITE,
    normalize_stored_token,
)


@pytest.fixture(autouse=True)
def _clear_security_env(monkeypatch):
    for name in (
        "SUMMARIZER_DEPLOY_MODE",
        "SUMMARIZER_API_TOKEN",
        "SUMMARIZER_API_TOKEN_SCOPES",
        "SUMMARIZER_OIDC_ISSUER",
        "SUMMARIZER_OIDC_AUDIENCE",
        "SUMMARIZER_OIDC_JWKS_URI",
        "SUMMARIZER_OIDC_ALGORITHMS",
        "SUMMARIZER_OIDC_SCOPES_CLAIM",
        "SUMMARIZER_OIDC_PROVIDERS_CLAIM",
        "SUMMARIZER_LOCAL_SOURCE_ROOTS",
        "SUMMARIZER_OUTBOUND_ALLOW_PRIVATE_ORIGINS",
        "SUMMARIZER_OUTBOUND_ALLOW_PRIVATE_CIDRS",
        "SUMMARIZER_OUTBOUND_ALLOWED_PORTS",
        "SUMMARIZER_CONFIRM_NON_LOOPBACK_BIND",
        "SUMMARIZER_CORS_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestModeResolution:
    def test_default_is_local_trusted(self):
        settings = load_server_settings(file_config={})
        assert settings.mode == MODE_LOCAL_TRUSTED

    def test_mode_from_yaml(self):
        settings = load_server_settings(
            file_config={
                "server": {
                    "mode": MODE_AUTHENTICATED_SERVER,
                    "auth": {"tokens": [{"id": "a", "token": "secret"}]},
                }
            }
        )
        assert settings.mode == MODE_AUTHENTICATED_SERVER
        assert len(settings.tokens) == 1

    def test_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_LOCAL_TRUSTED)
        settings = load_server_settings(
            file_config={"server": {"mode": MODE_AUTHENTICATED_SERVER}}
        )
        assert settings.mode == MODE_LOCAL_TRUSTED

    def test_cli_arg_overrides_env_and_yaml(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_LOCAL_TRUSTED)
        settings = load_server_settings(
            mode_arg=MODE_AUTHENTICATED_SERVER,
            file_config={
                "server": {
                    "mode": MODE_LOCAL_TRUSTED,
                    "auth": {"tokens": [{"id": "a", "token": "secret"}]},
                }
            },
        )
        assert settings.mode == MODE_AUTHENTICATED_SERVER

    def test_unknown_mode_rejected(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", "banana")
        with pytest.raises(ConfigurationError):
            load_server_settings(file_config={})

    def test_compatibility_arg_normalized_for_server_factory(self):
        # The server factory may never run compatibility; it falls back to
        # local-trusted (the CLI single-machine path never calls the factory).
        settings = load_server_settings(mode_arg="compatibility", file_config={})
        assert settings.mode == MODE_LOCAL_TRUSTED


class TestStrictFailFast:
    def test_strict_without_credentials_rejected(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        with pytest.raises(ConfigurationError) as exc:
            load_server_settings(file_config={})
        message = str(exc.value)
        assert "SUMMARIZER_API_TOKEN" in message
        assert "SUMMARIZER_OIDC_ISSUER" in message

    def test_strict_with_env_token_accepted(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        monkeypatch.setenv("SUMMARIZER_API_TOKEN", "raw-secret")
        settings = load_server_settings(file_config={})
        assert settings.is_strict and settings.auth_required
        assert settings.tokens[0].token_hash == hash_token("raw-secret")

    def test_strict_with_oidc_only_accepted(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        monkeypatch.setenv("SUMMARIZER_OIDC_ISSUER", "https://idp.example.com")
        monkeypatch.setenv("SUMMARIZER_OIDC_AUDIENCE", "summarize-api")
        settings = load_server_settings(file_config={})
        assert settings.oidc is not None
        assert settings.oidc.issuer == "https://idp.example.com"
        assert settings.oidc.audience == "summarize-api"

    def test_oidc_partial_env_rejected(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        monkeypatch.setenv("SUMMARIZER_OIDC_ISSUER", "https://idp.example.com")
        with pytest.raises(ConfigurationError):
            load_server_settings(file_config={})


class TestTokenNormalization:
    def test_plaintext_and_hash_equivalent(self):
        raw = "my-secret-token"
        assert normalize_stored_token(raw) == hash_token(raw)
        prehashed = "sha256:" + hash_token(raw)
        assert normalize_stored_token(prehashed) == hash_token(raw)

    def test_invalid_hash_rejected(self):
        with pytest.raises(ConfigurationError):
            normalize_stored_token("sha256:deadbeef")

    def test_empty_token_rejected(self):
        with pytest.raises(ConfigurationError):
            normalize_stored_token("   ")

    def test_yaml_token_requires_exactly_one_form(self):
        with pytest.raises(ConfigurationError):
            load_server_settings(
                file_config={
                    "server": {
                        "mode": MODE_AUTHENTICATED_SERVER,
                        "auth": {
                            "tokens": [
                                {"id": "a", "token": "x", "token_hash": "sha256:" + "a" * 64}
                            ]
                        },
                    }
                }
            )

    def test_env_token_default_scopes(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        monkeypatch.setenv("SUMMARIZER_API_TOKEN", "secret")
        settings = load_server_settings(file_config={})
        assert settings.tokens[0].scopes == frozenset(ALL_KNOWN_SCOPES)

    def test_env_token_explicit_scopes(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_DEPLOY_MODE", MODE_AUTHENTICATED_SERVER)
        monkeypatch.setenv("SUMMARIZER_API_TOKEN", "secret")
        monkeypatch.setenv("SUMMARIZER_API_TOKEN_SCOPES", SCOPE_WRITE)
        settings = load_server_settings(file_config={})
        assert settings.tokens[0].scopes == frozenset({SCOPE_WRITE})

    def test_unknown_configured_scope_rejected(self):
        with pytest.raises(ConfigurationError):
            load_server_settings(
                file_config={
                    "server": {
                        "mode": MODE_AUTHENTICATED_SERVER,
                        "auth": {
                            "tokens": [
                                {"id": "a", "token": "x", "scopes": ["not:a-scope"]}
                            ]
                        },
                    }
                }
            )

    def test_token_provider_restriction(self):
        settings = load_server_settings(
            file_config={
                "server": {
                    "mode": MODE_AUTHENTICATED_SERVER,
                    "auth": {
                        "tokens": [
                            {
                                "id": "ci",
                                "token": "x",
                                "providers": ["groq", "gemini"],
                            }
                        ]
                    },
                }
            }
        )
        assert settings.tokens[0].providers == frozenset({"groq", "gemini"})


class TestOutboundAndRootsConfig:
    def test_default_ports(self):
        settings = load_server_settings(file_config={})
        assert settings.allowed_ports == (80, 443)

    def test_ports_from_yaml(self):
        settings = load_server_settings(
            file_config={"outbound": {"allowed_ports": [80, 443, 8080]}}
        )
        assert settings.allowed_ports == (80, 443, 8080)

    def test_invalid_port_rejected(self):
        with pytest.raises(ConfigurationError):
            load_server_settings(file_config={"outbound": {"allowed_ports": [70000]}})

    def test_cidrs_validated(self):
        with pytest.raises(ConfigurationError):
            load_server_settings(
                file_config={"outbound": {"allow_private_cidrs": ["not-a-cidr"]}}
            )

    def test_private_origins_from_yaml_and_env(self, monkeypatch):
        settings = load_server_settings(
            file_config={"outbound": {"allow_private_origins": ["http://cobalt:9000"]}}
        )
        assert settings.allow_private_origins == ("http://cobalt:9000",)
        monkeypatch.setenv(
            "SUMMARIZER_OUTBOUND_ALLOW_PRIVATE_ORIGINS", "http://gw.internal:8080"
        )
        settings = load_server_settings(
            file_config={"outbound": {"allow_private_origins": ["http://cobalt:9000"]}}
        )
        assert settings.allow_private_origins == ("http://gw.internal:8080",)

    def test_local_roots_precedence(self, monkeypatch, tmp_path):
        yaml_root = tmp_path / "yaml-root"
        env_root = tmp_path / "env-root"
        cli_root = tmp_path / "cli-root"
        for path in (yaml_root, env_root, cli_root):
            path.mkdir()
        monkeypatch.setenv("SUMMARIZER_LOCAL_SOURCE_ROOTS", str(env_root))
        settings = load_server_settings(
            file_config={"server": {"local_source_roots": [str(yaml_root)]}}
        )
        assert settings.local_source_roots == (str(env_root),)
        settings = load_server_settings(
            local_source_roots_arg=[str(cli_root)],
            file_config={"server": {"local_source_roots": [str(yaml_root)]}},
        )
        assert settings.local_source_roots == (str(cli_root),)


class TestBindHostGuard:
    def test_loopback_allowed(self):
        settings = load_server_settings(file_config={})
        validate_bind_host("127.0.0.1", settings)
        validate_bind_host("localhost", settings)
        validate_bind_host("::1", settings)

    def test_non_loopback_local_trusted_rejected(self):
        settings = load_server_settings(file_config={})
        with pytest.raises(ConfigurationError) as exc:
            validate_bind_host("0.0.0.0", settings)
        assert "--confirm-non-loopback-bind" in str(exc.value)

    def test_non_loopback_with_confirmation_allowed(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_CONFIRM_NON_LOOPBACK_BIND", "true")
        settings = load_server_settings(file_config={})
        validate_bind_host("0.0.0.0", settings)

    def test_non_loopback_strict_allowed_without_confirmation(self):
        settings = load_server_settings(
            file_config={
                "server": {
                    "mode": MODE_AUTHENTICATED_SERVER,
                    "auth": {"tokens": [{"id": "a", "token": "x"}]},
                },
            }
        )
        validate_bind_host("0.0.0.0", settings)


# ── TR-9.x: `summarizer serve` CLI wiring ───────────────────────────────────


@pytest.fixture
def serve_guards(monkeypatch):
    """Block real config discovery and record uvicorn.run invocations."""
    from summarizer.security.socketguard import (
        socket_guard_active,
        uninstall_socket_guard,
    )

    monkeypatch.setattr(
        "summarizer.__main__.load_config_file", lambda *a, **k: {}
    )
    monkeypatch.setattr("summarizer.server.load_config_file", lambda: {})
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, host, port: calls.append((host, port))
    )
    yield calls
    while socket_guard_active():
        uninstall_socket_guard()


def _run_cli(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["summarizer"] + argv)
    from summarizer.__main__ import cli

    return cli


def test_tr9_1_strict_without_credentials_exits_nonzero(
    monkeypatch, capsys, serve_guards
):
    cli = _run_cli(["serve", "--mode", "authenticated-server"], monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        cli()
    assert exc_info.value.code == 1
    assert serve_guards == []
    output = capsys.readouterr().out
    assert "SUMMARIZER_API_TOKEN" in output or "authenticated-server" in output


def test_tr9_2_local_trusted_non_loopback_refused_then_confirmed(
    monkeypatch, capsys, serve_guards
):
    cli = _run_cli(["serve", "--host", "0.0.0.0"], monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        cli()
    assert exc_info.value.code == 1
    assert serve_guards == []
    assert "non-loopback" in capsys.readouterr().out

    cli = _run_cli(
        [
            "serve",
            "--host", "0.0.0.0",
            "--confirm-non-loopback-bind",
        ],
        monkeypatch,
    )
    cli()
    assert serve_guards == [("0.0.0.0", 8000)]


def test_tr9_2_strict_mode_starts_with_token(monkeypatch, serve_guards):
    monkeypatch.setenv("SUMMARIZER_API_TOKEN", "cli-token-value")
    cli = _run_cli(
        ["serve", "--mode", "authenticated-server", "--port", "9100"],
        monkeypatch,
    )
    cli()
    assert serve_guards == [("127.0.0.1", 9100)]


def test_tr9_3_standalone_cli_never_loads_server_settings(
    monkeypatch, serve_guards
):
    from summarizer.security.socketguard import socket_guard_active

    def _forbidden(*a, **k):
        raise AssertionError("standalone CLI must not load server settings")

    monkeypatch.setattr(
        "summarizer.security.settings.load_server_settings", _forbidden
    )
    cli = _run_cli(
        ["--source", "https://youtube.com/watch?v=x", "--no-config"],
        monkeypatch,
    )
    with pytest.raises(SystemExit):
        cli()  # exits on missing base_url/model, never on server settings
    assert socket_guard_active() is False
    assert serve_guards == []
