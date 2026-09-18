"""Integration tests for the process-wide socket guard against a loopback HTTPD."""

import asyncio
import http.server
import socket
import socketserver
import threading
import urllib.request

import pytest

from summarizer.security.netpolicy import (
    OutboundPolicy,
    OutboundPolicyError,
    reset_policy,
    set_policy,
)
from summarizer.security.socketguard import (
    install_socket_guard,
    socket_guard,
    socket_guard_active,
    uninstall_socket_guard,
)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def loopback_httpd():
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}/", f"http://localhost:{port}/"
    httpd.shutdown()
    httpd.server_close()


STRICT = OutboundPolicy(mode="authenticated-server", enforce=True)


def _strict_with_registered_localhost(port: int) -> OutboundPolicy:
    from summarizer.security.netpolicy import Origin, PURPOSE_COBALT_API

    origin = Origin("http", "localhost", port)
    return OutboundPolicy(
        mode="authenticated-server",
        enforce=True,
        exact_origins={PURPOSE_COBALT_API: {origin}},
        exempt_hosts={"localhost"},
    )


# ── install/uninstall bookkeeping ──────────────────────────────────────────


def test_install_is_reference_counted_and_restores():
    original = socket.getaddrinfo
    d1 = install_socket_guard()
    d2 = install_socket_guard()
    assert d1 == 1 and d2 == 2
    assert socket_guard_active() is True
    assert socket.getaddrinfo.__name__ == "_guarded_getaddrinfo"
    assert uninstall_socket_guard() == 1
    assert socket_guard_active() is True
    assert uninstall_socket_guard() == 0
    assert socket.getaddrinfo is original
    assert socket_guard_active() is False


def test_guard_context_manager(loopback_httpd):
    url, _ = loopback_httpd
    with socket_guard():
        assert socket_guard_active() is True
        # Default/permissive policy: loopback keeps working.
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200
    assert socket_guard_active() is False


# ── strict blocking across client stacks ────────────────────────────────────


def test_urllib_blocked(loopback_httpd):
    url, _ = loopback_httpd
    with socket_guard():
        token = set_policy(STRICT)
        try:
            with pytest.raises(OSError):
                urllib.request.urlopen(url, timeout=5)
        finally:
            from summarizer.security.netpolicy import reset_policy

            reset_policy(token)


def test_requests_blocked(loopback_httpd):
    import requests

    url, _ = loopback_httpd
    with socket_guard():
        token = set_policy(STRICT)
        try:
            with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
                requests.get(url, timeout=5, proxies={"http": None, "https": None})
            cause_chain = []
            exc = excinfo.value
            while exc is not None:
                cause_chain.append(type(exc))
                exc = exc.__cause__ or exc.__context__
            assert OutboundPolicyError in cause_chain
        finally:
            from summarizer.security.netpolicy import reset_policy

            reset_policy(token)


def test_raw_socket_blocked(loopback_httpd):
    url, _localhost_url = loopback_httpd
    port = int(url.rsplit(":", 1)[1].strip("/"))
    with socket_guard():
        token = set_policy(STRICT)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(OutboundPolicyError):
                sock.connect(("127.0.0.1", port))
            sock.close()
            sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            assert sock2.connect_ex(("127.0.0.1", port)) != 0
            sock2.close()
        finally:
            from summarizer.security.netpolicy import reset_policy

            reset_policy(token)


def test_gethostbyname_blocked():
    with socket_guard():
        token = set_policy(STRICT)
        try:
            with pytest.raises(OutboundPolicyError):
                socket.gethostbyname("localhost")
            with pytest.raises(OutboundPolicyError):
                socket.gethostbyname_ex("localhost")
        finally:
            from summarizer.security.netpolicy import reset_policy

            reset_policy(token)


def test_aiohttp_blocked(loopback_httpd):
    import aiohttp

    url, _ = loopback_httpd

    async def go():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientConnectorError) as excinfo:
                async with session.get(url, timeout=5) as resp:
                    await resp.text()
            exc = excinfo.value
            chain = []
            while exc is not None:
                chain.append(type(exc))
                exc = exc.__cause__ or exc.__context__
            assert OutboundPolicyError in chain

    with socket_guard():
        token = set_policy(STRICT)
        try:
            asyncio.run(go())
        finally:
            from summarizer.security.netpolicy import reset_policy

            reset_policy(token)


# ── registered infrastructure exemption ─────────────────────────────────────


def test_exempt_localhost_only_within_registered_purpose(loopback_httpd):
    import requests

    from summarizer.security.netpolicy import (
        PURPOSE_COBALT_API,
        resolution_scope,
    )

    _ip_url, localhost_url = loopback_httpd
    port = localhost_url.rsplit(":", 1)[1].rstrip("/")
    policy = _strict_with_registered_localhost(int(port))
    with socket_guard():
        token = set_policy(policy)
        try:
            # A user-style request outside an exact-purpose scope must NOT
            # ride the registered localhost hostname (SSRF).
            with pytest.raises(Exception):
                requests.get(
                    localhost_url,
                    timeout=5,
                    proxies={"http": None, "https": None},
                )
            # The guarded Cobalt call stack installs a purpose scope: the
            # same registered origin is then reachable.
            with resolution_scope(PURPOSE_COBALT_API):
                resp = requests.get(
                    localhost_url,
                    timeout=5,
                    proxies={"http": None, "https": None},
                )
            assert resp.status_code == 200
        finally:
            reset_policy(token)


def test_pin_cannot_be_reused_by_raw_ip_connect(loopback_httpd):
    """A fresh hostname pin must not authorize a raw IP-literal connect."""
    from summarizer.security.netpolicy import (
        PURPOSE_COBALT_API,
        OutboundPolicyError as _PolicyError,
        resolution_scope,
    )

    _ip_url, localhost_url = loopback_httpd
    port = int(localhost_url.rsplit(":", 1)[1].rstrip("/"))
    policy = _strict_with_registered_localhost(port)
    with socket_guard():
        token = set_policy(policy)
        try:
            # Create a legitimate pin inside the exact-purpose scope.
            with resolution_scope(PURPOSE_COBALT_API):
                socket.getaddrinfo("localhost", port)
                # Outside the scope, the same IP:port must not ride the pin.
            with pytest.raises((_PolicyError, OSError)):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(3)
                    sock.connect(("127.0.0.1", port))
        finally:
            reset_policy(token)


def test_fallback_policy_enforced_in_worker_thread():
    """Libraries spawning raw worker threads still get strict enforcement."""
    errors = []

    def worker():
        try:
            # No per-thread ContextVar policy: fallback must apply.
            socket.getaddrinfo("localhost", 9000)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    install_socket_guard(fallback_policy=STRICT)
    try:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
    finally:
        uninstall_socket_guard()
    assert errors, "loopback resolution in an unscoped worker thread was allowed"
