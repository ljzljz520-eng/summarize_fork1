"""TR-10.2 strict deployment end-to-end drill.

Four assertion points exercised against one authenticated-server app:

1. token authentication on a protected endpoint;
2. a registered exact-origin (local Cobalt stand-in) is reachable through the
   app's own outbound policy despite resolving to loopback;
3. a redirect hop to a loopback IP literal is refused and never contacted;
4. local path sources are denied outside policy and allowed inside a root.
"""

import http.server
import threading

import pytest
from fastapi.testclient import TestClient

from summarizer.security.httpguards import guarded_request, session_for
from summarizer.security.netpolicy import (
    OutboundPolicyError,
    PURPOSE_COBALT_API,
    PURPOSE_COBALT_DOWNLOAD,
    reset_policy,
    set_policy,
)
from summarizer.security.settings import (
    ALL_KNOWN_SCOPES,
    MODE_AUTHENTICATED_SERVER,
    SCOPE_READ,
    SCOPE_WRITE,
    ServerSettings,
    TokenEntry,
    hash_token,
)
from summarizer.security.socketguard import (
    socket_guard_active,
    uninstall_socket_guard,
)
from summarizer.server import create_app

TOKEN = "drill-secret-token"


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence test output
        pass


def _start_server(handler_cls):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class _CobaltHandler(_QuietHandler):
    evil_port = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # First hop is the registered hostname; redirect to a loopback IP
        # literal, which must never get the hostname exemption.
        self.send_response(302)
        self.send_header("Location", f"http://127.0.0.1:{self.evil_port}/evil")
        self.send_header("Content-Length", "0")
        self.end_headers()


class _EvilHandler(_QuietHandler):
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        body = b"internal metadata"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _chain_contains(exc, expected):
    seen = {exc}
    current = exc
    while current.__cause__ is not None or current.__context__ is not None:
        nxt = current.__cause__ or current.__context__
        if nxt in seen:
            break
        seen.add(nxt)
        current = nxt
    return any(isinstance(item, expected) for item in seen)


@pytest.fixture
def drill(monkeypatch, tmp_path):
    cobalt_server, _ = _start_server(_CobaltHandler)
    evil_server, _ = _start_server(_EvilHandler)
    _CobaltHandler.evil_port = evil_server.server_address[1]
    _EvilHandler.hits = 0

    cobalt_port = cobalt_server.server_address[1]
    file_config = {
        "default_provider": "local",
        "providers": {
            "local": {
                # https-only model origins: kept non-routable here; main is
                # mocked for the local-source leg of the drill.
                "base_url": f"http://localhost:{cobalt_port}",
                "model": "drill-model",
            },
        },
        "defaults": {"cobalt-base-url": f"http://localhost:{cobalt_port}"},
    }
    monkeypatch.setattr(
        "summarizer.server.load_config_file", lambda: dict(file_config)
    )
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(
            TokenEntry(
                id="drill",
                token_hash=hash_token(TOKEN),
                scopes=frozenset(ALL_KNOWN_SCOPES),
            ),
        ),
        local_source_roots=(str(tmp_path),),
    )
    application = create_app(settings, file_config=file_config)
    client = TestClient(application)

    yield {
        "client": client,
        "app": application,
        "policy": application.state.outbound_policy,
        "cobalt_port": cobalt_port,
        "evil_server": evil_server,
        "tmp_path": tmp_path,
    }

    cobalt_server.shutdown()
    evil_server.shutdown()
    while socket_guard_active():
        uninstall_socket_guard()


def test_tr10_2_strict_drill(drill, monkeypatch):
    client = drill["client"]
    policy = drill["policy"]
    auth = {"Authorization": f"Bearer {TOKEN}"}

    # (1) Protected endpoint: 401 without credentials, 200 with token.
    assert client.get("/providers").status_code == 401
    response = client.get("/providers", headers=auth)
    assert response.status_code == 200
    assert {item["name"] for item in response.json()} == {"local"}

    # (2) Registered exact-origin (loopback Cobalt stand-in) is reachable via
    # the app's strict policy: hostname exemption pinned its resolved IP.
    token_ctx = set_policy(policy)
    try:
        response = guarded_request(
            "post",
            f"http://localhost:{drill['cobalt_port']}/api",
            policy=policy,
            purpose=PURPOSE_COBALT_API,
            json={"url": "https://youtube.com/watch?v=x"},
            timeout=10,
        )
        assert response.status_code == 200

        # (3) Redirect hop to a loopback IP literal: refused, zero contact.
        session = session_for(policy, PURPOSE_COBALT_DOWNLOAD)
        with pytest.raises(Exception) as exc_info:
            session.get(
                f"http://localhost:{drill['cobalt_port']}/redir",
                timeout=10,
            )
        assert _chain_contains(exc_info.value, OutboundPolicyError)
    finally:
        reset_policy(token_ctx)
    assert drill["evil_server"].RequestHandlerClass.hits == 0

    # (4a) Local path outside the configured root: 403.
    response = client.post(
        "/summarize",
        headers=auth,
        json={
            "source": "/etc/passwd",
            "type": "Local File",
            "provider": "local",
        },
    )
    assert response.status_code == 403

    # (4b) Local path inside the root: authorized through to main.
    captured = {}
    monkeypatch.setattr(
        "summarizer.server.main",
        lambda config: captured.update(config) or "drill summary",
    )
    media = drill["tmp_path"] / "clip.mp4"
    media.write_text("x")
    response = client.post(
        "/summarize",
        headers=auth,
        json={
            "source": str(media),
            "type": "Local File",
            "provider": "local",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["source_url_or_path"] == str(media.resolve())
    assert captured["base_url"] == f"http://localhost:{drill['cobalt_port']}"


def test_tr10_2_read_only_token_cannot_summarize(drill):
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(
            TokenEntry(
                id="ro",
                token_hash=hash_token(TOKEN),
                scopes=frozenset({SCOPE_READ}),
            ),
        ),
    )
    application = create_app(settings, file_config={
        "providers": {"local": {"base_url": "https://example.com/v1", "model": "m"}},
    })
    try:
        ro_client = TestClient(application)
        response = ro_client.post(
            "/summarize",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"source": "https://youtube.com/watch?v=x", "provider": "local"},
        )
        assert response.status_code == 403
    finally:
        while socket_guard_active():
            uninstall_socket_guard()
