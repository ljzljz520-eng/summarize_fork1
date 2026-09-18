"""Unit tests for origin normalization, IP classification and URL policy."""

import socket

import pytest

from summarizer.security.netpolicy import (
    HOST_SUFFIXES,
    Origin,
    OutboundPolicy,
    OutboundPolicyError,
    PURPOSE_COBALT_API,
    PURPOSE_DRIVE,
    PURPOSE_DROPBOX,
    PURPOSE_GENERIC_DOWNLOAD,
    PURPOSE_MODEL,
    PURPOSE_VISION,
    PURPOSE_YOUTUBE_CAPTIONS,
    build_outbound_policy,
    classify_ip,
    host_matches_suffix,
    is_public_ip,
    normalize_origin,
)
from summarizer.security.settings import (
    MODE_AUTHENTICATED_SERVER,
    MODE_LOCAL_TRUSTED,
    ServerSettings,
    TokenEntry,
)


# ── TR-2.1: origin normalization ────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "HTTPS://API.Example.COM:443/openai/v1/chat/completions?q=1",
            Origin("https", "api.example.com", 443),
        ),
        ("https://api.example.com/", Origin("https", "api.example.com", 443)),
        ("http://example.com:80/path", Origin("http", "example.com", 80)),
        ("https://example.com:8443/x", Origin("https", "example.com", 8443)),
        ("http://bücher.example/", Origin("http", "xn--bcher-kva.example", 80)),
        ("https://例え.jp/", Origin("https", "xn--r8jz45g.jp", 443)),
    ],
)
def test_normalize_origin_valid(url, expected):
    assert normalize_origin(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "ftp://example.com/file",
        "http://user:pass@example.com/",
        "http://example.com:99999/",
        "http:///",
        "gopher://example.com",
    ],
)
def test_normalize_origin_rejects_invalid(url):
    with pytest.raises(OutboundPolicyError):
        normalize_origin(url)


def test_normalize_origin_rejects_ip_literal_by_default():
    with pytest.raises(OutboundPolicyError):
        normalize_origin("http://8.8.8.8/x")
    allowed = normalize_origin("http://8.8.8.8/x", allow_ip_literal=True)
    assert allowed.host == "8.8.8.8"


def test_host_matches_suffix():
    assert host_matches_suffix("drive.google.com", "google.com")
    assert host_matches_suffix("foo.googleusercontent.com", "googleusercontent.com")
    assert not host_matches_suffix("evilgoogle.com", "google.com")


# ── TR-2.2: IP classification ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "ip,category",
    [
        ("127.0.0.1", "loopback"),
        ("127.255.255.255", "loopback"),
        ("::1", "loopback"),
        ("::ffff:127.0.0.1", "loopback"),
        ("169.254.169.254", "link_local"),
        ("fe80::1", "link_local"),
        ("10.0.0.1", "private"),
        ("172.16.0.1", "private"),
        ("172.31.255.255", "private"),
        ("192.168.1.1", "private"),
        ("100.64.0.1", "private"),
        ("fc00::1", "private"),
        ("0.0.0.0", "unspecified"),
        ("::", "unspecified"),
        ("224.0.0.1", "multicast"),
        ("8.8.8.8", "global"),
        ("1.1.1.1", "global"),
        ("2606:4700::1", "global"),
    ],
)
def test_classify_ip(ip, category):
    assert classify_ip(ip) == category


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "::1",
        "169.254.169.254",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.0.1",
        "fc00::1",
        "0.0.0.0",
    ],
)
def test_strict_policy_rejects_dangerous_ips(ip):
    policy = OutboundPolicy(mode=MODE_AUTHENTICATED_SERVER, enforce=True)
    with pytest.raises(OutboundPolicyError):
        policy.check_endpoint_ip(ip, host="evil.example")


def test_strict_policy_allows_public_ip():
    policy = OutboundPolicy(mode=MODE_AUTHENTICATED_SERVER, enforce=True)
    policy.check_endpoint_ip("8.8.8.8")
    policy.check_endpoint_ip("2606:4700::1")
    assert is_public_ip("8.8.8.8")


def test_boundary_172_32_is_global():
    assert classify_ip("172.32.0.1") == "global"


# ── TR-2.3: mixed DNS results rejected ──────────────────────────────────────


def _fake_addrinfo(ips):
    results = []
    for ip in ips:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        results.append((family, socket.SOCK_STREAM, 6, "", (ip, 443, 0, 0) if ":" in ip else (ip, 443)))
    return results


def test_resolve_and_check_rejects_mixed_results(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: _fake_addrinfo(["8.8.8.8", "127.0.0.1"]),
    )
    policy = OutboundPolicy(mode=MODE_AUTHENTICATED_SERVER, enforce=True)
    with pytest.raises(OutboundPolicyError):
        policy.resolve_and_check("rebind.example")


def test_resolve_and_check_all_public(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: _fake_addrinfo(["8.8.8.8", "1.1.1.1"]),
    )
    policy = OutboundPolicy(mode=MODE_AUTHENTICATED_SERVER, enforce=True)
    assert policy.resolve_and_check("cdn.example") == ["8.8.8.8", "1.1.1.1"]


# ── TR-2.4: purpose x host/port matrix ──────────────────────────────────────


MODEL_ORIGIN = Origin("https", "api.example.com", 443)
COBALT_ORIGIN = Origin("http", "cobalt", 9000)


def strict_policy(allowed_ports=(80, 443)):
    return OutboundPolicy(
        mode=MODE_AUTHENTICATED_SERVER,
        enforce=True,
        exact_origins={
            PURPOSE_MODEL: {MODEL_ORIGIN},
            PURPOSE_VISION: {MODEL_ORIGIN},
            PURPOSE_COBALT_API: {COBALT_ORIGIN},
        },
        allowed_ports=set(allowed_ports),
    )


def test_model_purpose_exact_origin_and_https():
    policy = strict_policy()
    policy.check_url("https://api.example.com/openai/v1/chat/completions", PURPOSE_MODEL)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("https://evil.example.com/v1/chat", PURPOSE_MODEL)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("http://api.example.com/v1/chat", PURPOSE_MODEL)


def test_drive_purpose_suffix_allowlist():
    policy = strict_policy()
    for url in (
        "https://drive.google.com/uc?export=download&id=1",
        "https://docs.google.com/document/x",
        "https://drive.usercontent.google.com/download?id=1",
        "https://foo.googleusercontent.com/doc",
    ):
        policy.check_url(url, PURPOSE_DRIVE)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("https://evil.example.com/virus", PURPOSE_DRIVE)


def test_dropbox_purpose_suffix_allowlist():
    policy = strict_policy()
    policy.check_url("https://www.dropbox.com/s/abc/v?dl=1", PURPOSE_DROPBOX)
    policy.check_url("https://dl.dropboxusercontent.com/1/v.mp4", PURPOSE_DROPBOX)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("https://evil.example.com/v", PURPOSE_DROPBOX)


def test_youtube_captions_suffix_allowlist():
    policy = strict_policy()
    policy.check_url("https://www.youtube.com/watch?v=x", PURPOSE_YOUTUBE_CAPTIONS)
    policy.check_url("https://rr3---sn-x.googlevideo.com/videoplayback", PURPOSE_YOUTUBE_CAPTIONS)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("https://evil.example.com/v", PURPOSE_YOUTUBE_CAPTIONS)


def test_cobalt_api_exact_origin_any_port():
    policy = strict_policy()
    policy.check_url("http://cobalt:9000/", PURPOSE_COBALT_API)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("http://cobalt:80/", PURPOSE_COBALT_API)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("http://evil:9000/", PURPOSE_COBALT_API)


def test_generic_download_public_hosts_and_ports():
    policy = strict_policy()
    policy.check_url("https://cdn.example.com/video.mp4", PURPOSE_GENERIC_DOWNLOAD)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("https://cdn.example.com:8080/v", PURPOSE_GENERIC_DOWNLOAD)
    policy2 = strict_policy(allowed_ports=(80, 443, 8080))
    policy2.check_url("https://cdn.example.com:8080/v", PURPOSE_GENERIC_DOWNLOAD)


def test_generic_download_ip_literal_urls():
    policy = strict_policy()
    policy.check_url("http://8.8.8.8/v.mp4", PURPOSE_GENERIC_DOWNLOAD)
    with pytest.raises(OutboundPolicyError):
        policy.check_url("http://127.0.0.1/v.mp4", PURPOSE_GENERIC_DOWNLOAD)


def test_permissive_policy_allows_private():
    policy = OutboundPolicy(mode=MODE_LOCAL_TRUSTED, enforce=False)
    policy.check_url("http://localhost:9000/", PURPOSE_COBALT_API)
    policy.check_endpoint_ip("127.0.0.1")


# ── build_outbound_policy ───────────────────────────────────────────────────


def _strict_settings():
    return ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        tokens=(TokenEntry(id="a", token_hash="x", scopes=frozenset({"summarize:write"})),),
        allow_private_origins=("http://gw.internal:8080",),
        allow_private_cidrs=("10.0.0.0/8",),
        allowed_ports=(80, 443),
    )


def test_build_strict_policy_exempts_infra(monkeypatch):
    # Avoid real DNS resolution during preload.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _fake_addrinfo(["10.0.0.5"]))
    policy = build_outbound_policy(
        _strict_settings(),
        provider_origins=[("groq", MODEL_ORIGIN)],
        cobalt_origin=COBALT_ORIGIN,
        proxy_endpoints=[("proxy.local", 3128)],
    )
    assert policy.enforce is True
    assert MODEL_ORIGIN in policy.exact_origins[PURPOSE_MODEL]
    assert COBALT_ORIGIN in policy.exact_origins[PURPOSE_COBALT_API]
    assert {"api.example.com", "cobalt", "gw.internal", "proxy.local"} <= policy.exempt_hosts
    # Exempt infra host with private IP is accepted.
    policy.check_endpoint_ip("10.0.0.9", host="gw.internal")


def test_proxy_exemption_bound_to_registered_port():
    policy = OutboundPolicy(
        mode=MODE_AUTHENTICATED_SERVER,
        enforce=True,
        proxy_endpoints={"proxy.local": {3128}},
    )
    # The registered proxy endpoint is exempt for every purpose and gets a pin.
    policy.check_endpoint_ip(
        "10.0.0.5",
        host="proxy.local",
        port=3128,
        purpose=PURPOSE_GENERIC_DOWNLOAD,
    )
    policy.check_connect_ip("10.0.0.5", host="proxy.local", port=3128)
    # The same hostname never exempts another port (no service probing ride).
    with pytest.raises(OutboundPolicyError):
        policy.check_endpoint_ip(
            "10.0.0.5",
            host="proxy.local",
            port=5432,
            purpose=PURPOSE_GENERIC_DOWNLOAD,
        )
    with pytest.raises(OutboundPolicyError):
        policy.check_connect_ip("10.0.0.5", host="proxy.local", port=5432)
    # No exemption without a registered port at all.
    with pytest.raises(OutboundPolicyError):
        policy.check_endpoint_ip(
            "10.0.0.6", host="proxy.local", purpose=PURPOSE_GENERIC_DOWNLOAD
        )


def test_build_local_trusted_is_permissive():
    policy = build_outbound_policy(ServerSettings(mode=MODE_LOCAL_TRUSTED))
    assert policy.enforce is False
