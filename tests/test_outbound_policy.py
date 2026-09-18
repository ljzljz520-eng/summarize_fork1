"""Tests for policy-guarded requests/aiohttp clients (redirect-hop checks)."""

import asyncio
import http.server
import io
import socketserver
import threading

import pytest

from summarizer.security.netpolicy import (
    Origin,
    OutboundPolicy,
    OutboundPolicyError,
    PURPOSE_COBALT_API,
    PURPOSE_GENERIC_DOWNLOAD,
    PURPOSE_MODEL,
    reset_policy,
    set_policy,
)
from summarizer.security.httpguards import (
    _build_aiohttp_connector,
    guarded_aiohttp_session,
    session_for,
)
from summarizer.security.socketguard import socket_guard


class _RedirectHandler(http.server.BaseHTTPRequestHandler):
    """GET /redirect -> 302 target; GET /metadata -> 302 link-local; else 200."""

    protocol_version = "HTTP/1.1"
    target_url = ""
    hits = None

    def _bump(self):
        if self.hits is not None:
            self.hits[self.path] = self.hits.get(self.path, 0) + 1

    def _reply(self, status, body=b"", headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._bump()
        if self.path == "/redirect":
            self._reply(302, headers={"Location": self.target_url})
        elif self.path == "/metadata":
            self._reply(
                302,
                headers={"Location": "http://169.254.169.254/latest/meta-data"},
            )
        else:
            self._reply(200, b"ok", {"Content-Type": "text/plain"})

    def do_POST(self):  # noqa: N802
        self._bump()
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._reply(200, b'{"ok":true}', {"Content-Type": "application/json"})

    def log_message(self, *args):
        pass


def _assert_cause(exc_type, fn):
    with pytest.raises(Exception) as excinfo:
        fn()
    chain = []
    exc = excinfo.value
    while exc is not None:
        chain.append(type(exc))
        exc = exc.__cause__ or exc.__context__
    assert exc_type in chain, f"{exc_type.__name__} not in {chain}"


def _serve(target_url="", bind_ip="127.0.0.1"):
    hits = {}
    handler = type("_H", (_RedirectHandler,), {"target_url": target_url, "hits": hits})
    httpd = socketserver.ThreadingTCPServer((bind_ip, 0), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    return httpd, port, hits


@pytest.fixture
def two_servers():
    target, target_port, target_hits = _serve()
    start, start_port, start_hits = _serve(
        target_url=f"http://127.0.0.1:{target_port}/secret"
    )
    try:
        yield start_port, start_hits, target_port, target_hits
    finally:
        start.shutdown()
        start.server_close()
        target.shutdown()
        target.server_close()


@pytest.fixture
def strict_policy(two_servers):
    start_port, _sh, target_port, _th = two_servers
    cobalt_origin = Origin("http", "localhost", start_port)
    model_origin = Origin("https", "api.example.com", 443)
    policy = OutboundPolicy(
        mode="authenticated-server",
        enforce=True,
        exact_origins={
            PURPOSE_COBALT_API: {cobalt_origin},
            PURPOSE_MODEL: {model_origin},
        },
        allowed_ports={80, 443, start_port, target_port},
        exempt_hosts={"localhost", "start.test"},
        # Simulates a server-registered internal proxy endpoint; the redirect
        # tests resolve it to loopback via a DNS stand-in.
        proxy_endpoints={"start.test": {start_port}},
    )
    return policy


@pytest.fixture
def start_hostname_dns(monkeypatch):
    """Resolve the fake proxy hostname 'start.test' to loopback."""
    from summarizer.security import socketguard

    original_resolver = socketguard._ORIGINALS.get("getaddrinfo")
    if original_resolver is None:  # guard not installed yet; use stdlib
        import socket as _socket

        original_resolver = _socket.getaddrinfo

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        if host == "start.test":
            return original_resolver("127.0.0.1", port, *args, **kwargs)
        return original_resolver(host, port, *args, **kwargs)

    monkeypatch.setitem(socketguard._ORIGINALS, "getaddrinfo", _fake_getaddrinfo)


@pytest.fixture
def guarded_ctx(strict_policy):
    with socket_guard():
        token = set_policy(strict_policy)
        try:
            yield strict_policy
        finally:
            reset_policy(token)


_NO_PROXY = {"http": None, "https": None}


# ── TR-4.1: download redirect to internal IP blocked, target never hit ──────


def test_guarded_session_blocks_redirect_to_loopback(
    two_servers, guarded_ctx, start_hostname_dns
):
    start_port, start_hits, _target_port, target_hits = two_servers
    sess = session_for(guarded_ctx, PURPOSE_GENERIC_DOWNLOAD, trust_env=False)
    _assert_cause(
        OutboundPolicyError,
        lambda: sess.get(
            f"http://start.test:{start_port}/redirect",
            timeout=5,
            proxies=_NO_PROXY,
        ),
    )
    # The registered first-hop hostname is contacted once; its pin is bound to
    # (start.test, 127.0.0.1, port), so the 302 to an IP-literal loopback
    # target on another port is blocked hop-by-hop and never contacted.
    assert start_hits.get("/redirect", 0) == 1
    assert target_hits == {}


def test_guarded_session_blocks_redirect_to_link_local(
    two_servers, guarded_ctx, start_hostname_dns
):
    start_port, start_hits, _tp, target_hits = two_servers
    sess = session_for(guarded_ctx, PURPOSE_GENERIC_DOWNLOAD, trust_env=False)
    _assert_cause(
        OutboundPolicyError,
        lambda: sess.get(
            f"http://start.test:{start_port}/metadata",
            timeout=5,
            proxies=_NO_PROXY,
        ),
    )
    assert target_hits == {}


# ── TR-4.2: model sessions reject unbound origins and never follow 3xx ──────


def test_model_session_rejects_unbound_origin(guarded_ctx):
    sess = session_for(guarded_ctx, PURPOSE_MODEL)
    _assert_cause(
        OutboundPolicyError,
        lambda: sess.get("https://evil.example.com/v1/chat/completions", timeout=5),
    )


def test_model_session_does_not_follow_redirects(guarded_ctx, monkeypatch):
    import requests

    sess = session_for(guarded_ctx, PURPOSE_MODEL)
    url = "https://api.example.com/v1/chat/completions"
    prepared = requests.Request("GET", url).prepare()
    resp = requests.Response()
    resp.status_code = 302
    resp.headers["Location"] = "https://api.example.com/elsewhere"
    resp.raw = io.BytesIO(b"")
    resp.request = prepared

    calls = {"n": 0}

    def fake_send(request, *args, **kwargs):
        calls["n"] += 1
        return resp

    adapter = sess.get_adapter(url)
    monkeypatch.setattr(adapter, "send", fake_send)

    result = sess.get(url, allow_redirects=True, timeout=5)
    assert result.status_code == 302
    assert result.history == []
    assert calls["n"] == 1


# ── TR-4.3: registered cobalt origin allowed; same loopback as user URL not ─


def test_cobalt_registered_origin_allowed(two_servers, guarded_ctx):
    start_port, start_hits, _tp, target_hits = two_servers
    sess = session_for(guarded_ctx, PURPOSE_COBALT_API, trust_env=False)
    resp = sess.post(
        f"http://localhost:{start_port}/",
        json={"url": "x"},
        timeout=5,
        proxies=_NO_PROXY,
    )
    assert resp.status_code == 200
    assert start_hits.get("/", 0) == 1
    # cobalt_api never follows redirects: internal redirect target untouched.
    resp2 = sess.get(
        f"http://localhost:{start_port}/redirect",
        timeout=5,
        proxies=_NO_PROXY,
    )
    assert resp2.status_code == 302
    assert target_hits == {}


def test_same_loopback_as_generic_user_url_rejected(two_servers, guarded_ctx):
    start_port, _sh, _tp, _th = two_servers
    sess = session_for(guarded_ctx, PURPOSE_GENERIC_DOWNLOAD, trust_env=False)
    _assert_cause(
        OutboundPolicyError,
        lambda: sess.get(
            f"http://127.0.0.1:{start_port}/",
            timeout=5,
            proxies=_NO_PROXY,
        ),
    )


def test_exempt_hostname_cannot_be_ridden_by_user_url(two_servers, guarded_ctx):
    """H-1 regression: localhost is registered for cobalt_api, but a user
    supplied generic-download URL using the same hostname must be blocked."""
    start_port, start_hits, _tp, _th = two_servers
    sess = session_for(guarded_ctx, PURPOSE_GENERIC_DOWNLOAD, trust_env=False)
    _assert_cause(
        OutboundPolicyError,
        lambda: sess.get(
            f"http://localhost:{start_port}/",
            timeout=5,
            proxies=_NO_PROXY,
        ),
    )
    assert start_hits == {}


# ── Cobalt request body URL is itself policy-checked (open-relay guard) ─────


@pytest.mark.parametrize("user_url", [
    "http://127.0.0.1:9000/",
    "http://localhost:9000/",
    "http://169.254.169.254/latest/meta-data/",
])
def test_cobalt_body_url_preflight_blocks_internal(user_url):
    from summarizer.downloaders.cobalt import CobaltDownloader
    from summarizer.exceptions import AudioProcessingError
    from summarizer.security.netpolicy import reset_policy, set_policy

    policy = OutboundPolicy(mode="authenticated-server", enforce=True)
    token = set_policy(policy)
    try:
        downloader = CobaltDownloader("http://cobalt.example.test")
        with pytest.raises(AudioProcessingError) as excinfo:
            downloader.download_audio(user_url)
        chain = []
        exc = excinfo.value
        while exc is not None:
            chain.append(type(exc))
            exc = exc.__cause__ or exc.__context__
        assert OutboundPolicyError in chain
        # The user URL must not leak into the sanitized error message.
        assert "127.0.0.1" not in str(excinfo.value)
        assert "169.254.169.254" not in str(excinfo.value)
    finally:
        reset_policy(token)


def test_authorization_stripped_across_ports(guarded_ctx):
    sess = session_for(guarded_ctx, PURPOSE_MODEL)
    assert sess.should_strip_auth(
        "https://api.example.com/a", "https://api.example.com:8443/b"
    )
    assert sess.should_strip_auth(
        "https://api.example.com/a", "https://evil.example.com/b"
    )


# ── TR-4.4: guarded aiohttp connector ───────────────────────────────────────


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_guarded_connector_rejects_private_resolution(guarded_ctx, monkeypatch):
    import aiohttp

    addrs = [
        {"hostname": "rebind", "host": "8.8.8.8", "port": 443,
         "family": 2, "proto": 6, "flags": 0},
        {"hostname": "rebind", "host": "127.0.0.1", "port": 443,
         "family": 2, "proto": 6, "flags": 0},
    ]

    async def fake_resolve(self, *args, **kwargs):
        return addrs

    async def go():
        connector_cls = _build_aiohttp_connector()
        connector = connector_cls(policy=guarded_ctx)
        monkeypatch.setattr(aiohttp.TCPConnector, "_resolve_host", fake_resolve)
        try:
            with pytest.raises(OutboundPolicyError):
                await connector._resolve_host("rebind", 443)
        finally:
            await connector.close()

    _run(go())


def test_guarded_connector_allows_public_resolution(guarded_ctx, monkeypatch):
    import aiohttp

    addrs = [
        {"hostname": "cdn", "host": "8.8.8.8", "port": 443,
         "family": 2, "proto": 6, "flags": 0},
    ]

    async def fake_resolve(self, *args, **kwargs):
        return addrs

    async def go():
        connector_cls = _build_aiohttp_connector()
        connector = connector_cls(policy=guarded_ctx)
        monkeypatch.setattr(aiohttp.TCPConnector, "_resolve_host", fake_resolve)
        try:
            result = await connector._resolve_host("cdn", 443)
            assert result == addrs
        finally:
            await connector.close()

    _run(go())


def test_model_aiohttp_session_disables_redirects(guarded_ctx):
    async def go():
        session = guarded_aiohttp_session(guarded_ctx, PURPOSE_MODEL)
        try:
            assert session.guarded_forbid_redirects is True
        finally:
            await session.close()

    _run(go())


def test_download_aiohttp_session_allows_redirects(guarded_ctx):
    async def go():
        session = guarded_aiohttp_session(guarded_ctx, PURPOSE_GENERIC_DOWNLOAD)
        try:
            assert session.guarded_forbid_redirects is False
        finally:
            await session.close()

    _run(go())
