"""Unified outbound network policy (SSRF / DNS-rebinding protection).

Every outbound HTTP/HTTPS call made by the application must be authorized by an
:class:`OutboundPolicy` instance:

* ``check_url(url, purpose)`` validates scheme, port and host-level rules
  (exact-origin allowlists for model/vision/Cobalt endpoints, suffix allowlists
  for Google Drive/Dropbox/YouTube, arbitrary public hosts for generic video
  downloads).
* ``check_endpoint_ip(ip, host)`` validates every resolved IP address and is
  also invoked by the socket-level shims (see ``socketguard.py``) at actual
  connect time, closing the DNS-rebinding TOCTOU window.

A permissive policy is used by the ``compatibility`` and ``local-trusted``
modes so the standalone CLI keeps its historical behavior; the strict policy
is installed for ``authenticated-server``.
"""

import contextvars
import ipaddress
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .settings import (
    MODE_AUTHENTICATED_SERVER,
    MODE_COMPATIBILITY,
    MODE_LOCAL_TRUSTED,
    ServerSettings,
)

# ── Purposes ────────────────────────────────────────────────────────────────

PURPOSE_MODEL = "model"
PURPOSE_VISION = "vision"
PURPOSE_COBALT_API = "cobalt_api"
PURPOSE_COBALT_DOWNLOAD = "cobalt_download"
PURPOSE_GENERIC_DOWNLOAD = "generic_download"
PURPOSE_DRIVE = "drive"
PURPOSE_DROPBOX = "dropbox"
PURPOSE_YOUTUBE_CAPTIONS = "youtube_captions"
PURPOSE_OIDC = "oidc"

# Purposes whose host must equal a server-registered exact origin.
EXACT_ORIGIN_PURPOSES = frozenset(
    {PURPOSE_MODEL, PURPOSE_VISION, PURPOSE_COBALT_API, PURPOSE_OIDC}
)
# Purposes where every redirect/hop may target any public host.
PUBLIC_DOWNLOAD_PURPOSES = frozenset(
    {PURPOSE_COBALT_DOWNLOAD, PURPOSE_GENERIC_DOWNLOAD}
)
# Purposes constrained to a host-suffix allowlist.
SUFFIX_PURPOSES = frozenset(
    {PURPOSE_DRIVE, PURPOSE_DROPBOX, PURPOSE_YOUTUBE_CAPTIONS}
)
# Purposes that require HTTPS even in strict mode (model endpoints carry keys).
HTTPS_ONLY_PURPOSES = frozenset({PURPOSE_MODEL, PURPOSE_VISION})

HOST_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    PURPOSE_DRIVE: (
        "drive.google.com",
        "docs.google.com",
        "drive.usercontent.google.com",
        "googleusercontent.com",
        "ggpht.com",
    ),
    PURPOSE_DROPBOX: (
        "dropbox.com",
        "dropboxusercontent.com",
    ),
    PURPOSE_YOUTUBE_CAPTIONS: (
        "youtube.com",
        "youtu.be",
        "googlevideo.com",
        "ytimg.com",
    ),
}

DEFAULT_ALLOWED_PORTS = frozenset({80, 443})


class OutboundPolicyError(OSError):
    """Raised when an outbound request is rejected by the network policy.

    Subclasses OSError so HTTP clients surface it as a connection failure
    rather than attempting retries against the forbidden target.
    """


# ── Origin handling ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Origin:
    scheme: str
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def _idna_host(host: str) -> str:
    host = host.lower()
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise OutboundPolicyError(f"Host contains invalid IDNA characters: {host}") from exc


def normalize_origin(url: str, allow_ip_literal: bool = False) -> Origin:
    """Normalize a URL to an exact-match ``Origin``.

    Lowercases the scheme and host (IDNA), removes userinfo/path/query and
    elides default ports (80/443). Raises OutboundPolicyError for anything
    that is not an absolute http(s) URL. Registered origins never accept IP
    literals; generic public downloads may use a public IP literal.
    """
    if not isinstance(url, str) or not url.strip():
        raise OutboundPolicyError("URL must be a non-empty string.")
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise OutboundPolicyError(
            f"Only http/https URLs are allowed (got scheme {parsed.scheme!r})."
        )
    if parsed.username is not None or parsed.password is not None:
        raise OutboundPolicyError("URLs with embedded credentials are not allowed.")
    host = parsed.hostname
    if not host:
        raise OutboundPolicyError(f"URL is missing a host component: {url!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise OutboundPolicyError(f"URL contains an invalid port: {url!r}") from exc
    host = host.lower()
    if not _is_hostname(host):
        if not allow_ip_literal:
            raise OutboundPolicyError(
                f"IP-literal hosts are not allowed as registered origins: {host}"
            )
        normalized_host = host
    else:
        normalized_host = _idna_host(host)
    port = parsed.port or (443 if scheme == "https" else 80)
    return Origin(scheme=scheme, host=normalized_host, port=port)


def _is_hostname(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        return True


def _host_only(value: str) -> str:
    """Extract a bare lowercase hostname from 'host', 'host:port' or a URL."""
    value = (value or "").strip()
    if not value:
        return ""
    if "://" in value:
        parsed = urlparse(value)
        return (parsed.hostname or "").lower().rstrip(".")
    value = value.split("@")[-1].strip("[]").rstrip(".")
    if value.count(":") == 1 and value.rsplit(":", 1)[1].isdigit():
        value = value.rsplit(":", 1)[0]
    return value.lower()


def host_matches_suffix(host: str, suffix: str) -> bool:
    host = host.lower().rstrip(".")
    suffix = suffix.lower().rstrip(".")
    return host == suffix or host.endswith("." + suffix)


def origin_matches_url(origin: Origin, url: str) -> bool:
    """Exact-match a registered origin against a concrete request URL."""
    try:
        return normalize_origin(url) == origin
    except OutboundPolicyError:
        return False


# ── IP classification ──────────────────────────────────────────────────────


# Python 3.9 misses 100.64.0.0/10 (CGNAT) from is_private (fixed in 3.10).
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def classify_ip(ip: str) -> str:
    """Return a coarse category for an IP literal: global/loopback/..."""
    address = ipaddress.ip_address(ip)
    # Unwrap IPv4-mapped IPv6 (::ffff:a.b.c.d) which Python 3.9 mislabels as
    # 'reserved' instead of inheriting the IPv4 address properties.
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if address.is_unspecified:
        return "unspecified"
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link_local"
    if address.is_multicast:
        return "multicast"
    if address.is_private or address in _CGNAT_NETWORK:
        return "private"
    if address.is_reserved:
        return "reserved"
    return "global"


def is_public_ip(ip: str) -> bool:
    return classify_ip(ip) == "global"


def _normal_port(port) -> int:
    """Normalize a getaddrinfo/connect port (int or service name) to int."""
    if port is None:
        return 0
    try:
        return int(port)
    except (TypeError, ValueError):
        return 0


def resolve_all(host: str, port: Optional[int] = None) -> List[str]:
    """Resolve a host and return every returned IP (deduped, order preserved)."""
    results = socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP)
    ips: List[str] = []
    for family, _kind, _proto, _canon, sockaddr in results:
        ip = sockaddr[0]
        # Normalize IPv4-mapped IPv6 when reasonable; classify handles both.
        if ip not in ips:
            ips.append(ip)
    return ips


# ── Current-policy propagation ─────────────────────────────────────────────

# Default context policy is permissive (standalone CLI / Streamlit). The
# FastAPI strict app sets this per request; anyio copies the context into the
# worker threads used by run_in_threadpool.
_current_policy: "contextvars.ContextVar[Optional[OutboundPolicy]]" = contextvars.ContextVar(
    "summarizer_outbound_policy", default=None
)


def get_policy() -> "OutboundPolicy":
    policy = _current_policy.get()
    if policy is None:
        return _PERMISSIVE_POLICY
    return policy


def set_policy(policy: Optional["OutboundPolicy"]) -> contextvars.Token:
    return _current_policy.set(policy)


def reset_policy(token: contextvars.Token) -> None:
    _current_policy.reset(token)


# Resolution purpose for the current guarded call stack. A private-IP
# hostname exemption is granted ONLY while a guarded request for a matching
# exact-registered origin purpose is actively resolving/connecting; a user
# supplied download URL (generic/cobalt_download) therefore can never ride an
# infrastructure hostname exemption, even if it happens to share its name.
_current_resolution_purpose: "contextvars.ContextVar[Optional[str]]" = (
    contextvars.ContextVar("summarizer_resolution_purpose", default=None)
)


def current_resolution_purpose() -> Optional[str]:
    return _current_resolution_purpose.get()


@contextmanager
def resolution_scope(purpose: Optional[str]):
    token = _current_resolution_purpose.set(purpose)
    try:
        yield
    finally:
        _current_resolution_purpose.reset(token)


# ── Policy object ───────────────────────────────────────────────────────────


@dataclass
class OutboundPolicy:
    mode: str
    enforce: bool = True
    # purpose -> exact registered origins (model/vision/cobalt_api)
    exact_origins: Dict[str, Set[Origin]] = None
    # host suffix allowlists (constant per purpose; override only in tests)
    host_suffixes: Dict[str, Tuple[str, ...]] = None
    # ports allowed for non-exact / suffix purposes
    allowed_ports: Set[int] = None
    # hostnames registered as infrastructure (informational set / startup
    # logs); actual exemption decisions are purpose-scoped below.
    exempt_hosts: Set[str] = None
    # server-registered proxy endpoints: host -> ports. Private-IP exemption
    # applies for every purpose (the TCP connection itself targets the
    # proxy) but only on the exact registered port(s), so a user URL cannot
    # ride the proxy hostname to probe other services on the same address.
    proxy_endpoints: Dict[str, Set[int]] = None
    # CIDRs exempt from private-IP rejection
    exempt_cidrs: Tuple[ipaddress._BaseNetwork, ...] = ()
    # short-lived pins: (hostname, IP, port) -> timestamp. An endpoint is
    # pinned only when it was returned by DNS for a registered infrastructure
    # HOSTNAME inside a matching exact-purpose resolution scope (never for a
    # user-supplied IP literal); the connect hooks consult pins as the
    # DNS-rebinding backstop. Port is part of the key so a redirect to another
    # service on the same IP cannot ride the pin.
    _pinned_ips: Dict[Tuple[str, str, int], float] = None

    PIN_TTL = 60.0

    def __post_init__(self) -> None:
        if self.exact_origins is None:
            self.exact_origins = {}
        if self.host_suffixes is None:
            self.host_suffixes = dict(HOST_SUFFIXES)
        if self.allowed_ports is None:
            self.allowed_ports = set(DEFAULT_ALLOWED_PORTS)
        if self.exempt_hosts is None:
            self.exempt_hosts = set()
        if self.proxy_endpoints is None:
            self.proxy_endpoints = {}
        if self._pinned_ips is None:
            self._pinned_ips = {}
        self.exempt_hosts = {host.lower().rstrip(".") for host in self.exempt_hosts}
        self.proxy_endpoints = {
            host.lower().rstrip("."): {int(port) for port in ports}
            for host, ports in self.proxy_endpoints.items()
        }

    # ── URL layer ──

    def check_url(self, url: str, purpose: str) -> Origin:
        """Validate a concrete URL for a declared purpose. Returns its origin."""
        if not self.enforce:
            try:
                return normalize_origin(url, allow_ip_literal=True)
            except OutboundPolicyError:
                return Origin(scheme="", host="", port=0)
        origin = normalize_origin(
            url, allow_ip_literal=purpose in PUBLIC_DOWNLOAD_PURPOSES
        )
        # Public IP-literal hosts are validated through the IP layer too.
        if not _is_hostname(origin.host):
            if purpose not in PUBLIC_DOWNLOAD_PURPOSES:
                raise OutboundPolicyError(
                    f"{purpose} does not accept IP-literal hosts: {origin.host}"
                )
            self.check_endpoint_ip(origin.host)
        if purpose in EXACT_ORIGIN_PURPOSES:
            allowed = self.exact_origins.get(purpose, set())
            if origin not in allowed:
                expected = ", ".join(sorted(str(item) for item in allowed)) or "(none)"
                raise OutboundPolicyError(
                    f"{purpose} requests are bound to server-registered origins "
                    f"[{expected}], refusing {origin}."
                )
            if purpose in HTTPS_ONLY_PURPOSES and origin.scheme != "https":
                raise OutboundPolicyError(
                    f"{purpose} endpoints must use HTTPS, got {origin}."
                )
            return origin
        if purpose in SUFFIX_PURPOSES:
            suffixes = self.host_suffixes.get(purpose, ())
            if not any(host_matches_suffix(origin.host, suffix) for suffix in suffixes):
                raise OutboundPolicyError(
                    f"{purpose} requests are restricted to "
                    f"{', '.join(suffixes)}; refusing host {origin.host!r}."
                )
        elif purpose in PUBLIC_DOWNLOAD_PURPOSES:
            # Any host is acceptable at the URL layer; IP layer ensures public.
            pass
        else:
            raise OutboundPolicyError(f"Unknown outbound purpose: {purpose!r}")

        if origin.port not in self.allowed_ports:
            raise OutboundPolicyError(
                f"Port {origin.port} is not allowed for {purpose} "
                f"(allowed: {sorted(self.allowed_ports)}). Configure "
                "outbound.allowed_ports to extend."
            )
        return origin

    # ── IP layer ──

    def resolve_and_check(
        self,
        host: str,
        port: Optional[int] = None,
        *,
        purpose: Optional[str] = None,
    ) -> List[str]:
        """Resolve a host and validate EVERY returned address (DNS pin input)."""
        ips = resolve_all(host, port)
        if not ips:
            raise OutboundPolicyError(f"DNS returned no addresses for {host!r}.")
        for ip in ips:
            self.check_endpoint_ip(
                ip, host=host, port=port, purpose=purpose
            )
        return ips

    def _ip_in_exempt_cidr(self, ip: str) -> bool:
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            return False
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        for network in self.exempt_cidrs:
            if address in network:
                return True
        return False

    def _host_exempt_for_purpose(
        self, host: str, purpose: Optional[str], port: Optional[int] = None
    ) -> bool:
        """Private-IP hostname exemption is purpose-bound.

        Granted only for a hostname:port that is a registered proxy endpoint
        (the TCP target is the proxy itself for every purpose) or a hostname
        that exactly matches an origin registered for the *current exact
        purpose*. A user-supplied download URL can therefore never share an
        infrastructure exemption, and a proxy hostname cannot be used to
        reach arbitrary other ports on the proxy address.
        """
        if not host or not _is_hostname(host):
            return False
        proxy_ports = self.proxy_endpoints.get(host)
        if proxy_ports is not None and _normal_port(port) in proxy_ports:
            return True
        if purpose in EXACT_ORIGIN_PURPOSES:
            return any(
                origin.host == host
                for origin in self.exact_origins.get(purpose, set())
            )
        return False

    def check_endpoint_ip(
        self,
        ip: str,
        host: Optional[str] = None,
        *,
        port: Optional[int] = None,
        purpose: Optional[str] = "use-scope",
    ) -> None:
        """Validate a resolved address (called for EVERY DNS result).

        A non-public address is accepted only when it lies in a configured
        exempt CIDR, or when it was resolved through a registered
        infrastructure *hostname* inside a matching exact-purpose resolution
        scope. User-supplied IP literals never match a hostname exemption, and
        download purposes never receive hostname exemptions: an attacker cannot
        ride a registered host's name (e.g. ``localhost``) to reach loopback.
        """
        if not self.enforce:
            return
        category = classify_ip(ip)
        if category == "global":
            return
        if self._ip_in_exempt_cidr(ip):
            return
        normalized_host = (host or "").lower().rstrip(".")
        effective_purpose = (
            current_resolution_purpose() if purpose == "use-scope" else purpose
        )
        normalized_port = _normal_port(port)
        if self._host_exempt_for_purpose(
            normalized_host, effective_purpose, normalized_port
        ):
            self._pin(normalized_host, ip, normalized_port)
            return
        raise OutboundPolicyError(
            f"Refusing connection to {ip}"
            + (f" ({normalized_host})" if normalized_host else "")
            + f": address category {category!r} is blocked by the outbound policy "
            "(loopback/link-local/private/unspecified/multicast/reserved are denied; "
            "register server infrastructure via outbound.allow_private_origins "
            "or outbound.allow_private_cidrs)."
        )

    def check_connect_ip(
        self, ip: str, host: Optional[str] = None, *, port=None
    ) -> None:
        """Validate the actual connect() target (DNS-rebinding backstop).

        In addition to :meth:`check_endpoint_ip`, accepts endpoints that were
        pinned by a guarded resolution of the SAME registered hostname and
        port moments earlier. Pins never match across hostnames or ports, and
        an IP-literal connect is only accepted inside an active exact-purpose
        resolution scope whose registered origin owns the pin (a registered
        proxy pin is accepted for every purpose because the proxy itself is
        the TCP target).
        """
        if not self.enforce:
            return
        normalized_port = _normal_port(port)
        try:
            self.check_endpoint_ip(ip, host=host, port=normalized_port)
            return
        except OutboundPolicyError as denied:
            normalized_host = (host or "").lower().rstrip(".")
            if (
                normalized_host
                and _is_hostname(normalized_host)
                and self.is_pinned(normalized_host, ip, normalized_port)
            ):
                return
            purpose = current_resolution_purpose()
            allowed_hosts: Set[str] = set()
            if purpose in EXACT_ORIGIN_PURPOSES:
                allowed_hosts |= {
                    origin.host
                    for origin in self.exact_origins.get(purpose, set())
                }
            # A server-registered proxy is the real TCP target regardless of
            # the request purpose, so its fresh pins are acceptable; the pin
            # key already restricts this to the registered proxy port.
            allowed_hosts |= set(self.proxy_endpoints)
            if any(
                self.is_pinned(name, ip, normalized_port)
                for name in allowed_hosts
            ):
                return
            raise denied

    def _pin(self, host: str, ip: str, port: int) -> None:
        now = time.time()
        stale = [
            key
            for key, ts in self._pinned_ips.items()
            if now - ts > self.PIN_TTL
        ]
        for key in stale:
            self._pinned_ips.pop(key, None)
        self._pinned_ips[(host, ip, port)] = now

    def is_pinned(self, host: str, ip: str, port: int) -> bool:
        ts = self._pinned_ips.get((host, ip, port))
        return ts is not None and time.time() - ts <= self.PIN_TTL

    def preload_exempt_host(self, host: str, purpose: str, port: int = 0) -> None:
        """Resolve a registered infrastructure hostname eagerly (best effort).

        Only meaningful for an exact purpose whose registered origin set
        contains the host; never raises.
        """
        normalized = host.lower().rstrip(".")
        if purpose not in EXACT_ORIGIN_PURPOSES:
            return
        if not any(
            origin.host == normalized
            for origin in self.exact_origins.get(purpose, set())
        ):
            return
        try:
            with resolution_scope(purpose):
                for ip in resolve_all(normalized, port):
                    # Registered infrastructure may resolve to private IPs.
                    self.check_endpoint_ip(
                        ip,
                        host=normalized,
                        port=port,
                        purpose=purpose,
                    )
        except OSError:
            # The host may only resolve inside a later network namespace; the
            # getaddrinfo shim/connector pins it on first real resolution.
            pass


_PERMISSIVE_POLICY = OutboundPolicy(mode=MODE_COMPATIBILITY, enforce=False)


def build_outbound_policy(
    settings: ServerSettings,
    *,
    provider_origins: Optional[Iterable[Tuple[str, Origin]]] = None,
    cobalt_origin: Optional[Origin] = None,
    oidc_origin: Optional[Origin] = None,
    oidc_extra_origins: Optional[Iterable[Origin]] = None,
    proxy_endpoints: Optional[Iterable[Tuple[str, int]]] = None,
) -> OutboundPolicy:
    """Construct the OutboundPolicy for a resolved ServerSettings.

    Registered infrastructure (model origins, Cobalt origin, OIDC issuer,
    configured proxy hosts and ``outbound.allow_private_origins``) is bound to
    exact-purpose origin sets; private-IP exemption is only granted inside a
    matching purpose resolution scope. User-supplied URLs are never added to
    any exemption set.
    """
    if settings.mode != MODE_AUTHENTICATED_SERVER:
        return OutboundPolicy(
            mode=settings.mode if settings.mode in (MODE_LOCAL_TRUSTED, MODE_COMPATIBILITY)
            else MODE_LOCAL_TRUSTED,
            enforce=False,
        )

    exact_origins: Dict[str, Set[Origin]] = {
        PURPOSE_MODEL: set(),
        PURPOSE_VISION: set(),
        PURPOSE_COBALT_API: set(),
        PURPOSE_OIDC: set(),
    }
    infrastructure_hosts: Set[str] = set()
    registered_proxy_endpoints: Dict[str, Set[int]] = {}

    for _label, origin in provider_origins or []:
        exact_origins[PURPOSE_MODEL].add(origin)
        exact_origins[PURPOSE_VISION].add(origin)
        infrastructure_hosts.add(origin.host)
    if cobalt_origin is not None:
        exact_origins[PURPOSE_COBALT_API].add(cobalt_origin)
        infrastructure_hosts.add(cobalt_origin.host)
    if oidc_origin is not None:
        exact_origins[PURPOSE_OIDC].add(oidc_origin)
        infrastructure_hosts.add(oidc_origin.host)
    for origin in oidc_extra_origins or []:
        # Explicitly configured cross-origin JWKS endpoints.
        exact_origins[PURPOSE_OIDC].add(origin)
        infrastructure_hosts.add(origin.host)
    for raw in settings.allow_private_origins:
        # Administrator-registered private infrastructure: bind to the
        # server-initiated exact purposes (never to user download purposes).
        origin = normalize_origin(raw)
        exact_origins[PURPOSE_MODEL].add(origin)
        exact_origins[PURPOSE_VISION].add(origin)
        exact_origins[PURPOSE_COBALT_API].add(origin)
        infrastructure_hosts.add(origin.host)
    for host, port in proxy_endpoints or []:
        normalized_host = _host_only(host)
        if normalized_host:
            registered_proxy_endpoints.setdefault(
                normalized_host, set()
            ).add(int(port))
            infrastructure_hosts.add(normalized_host)

    networks: List[ipaddress._BaseNetwork] = []
    for raw in settings.allow_private_cidrs:
        networks.append(ipaddress.ip_network(raw, strict=False))

    policy = OutboundPolicy(
        mode=MODE_AUTHENTICATED_SERVER,
        enforce=True,
        exact_origins=exact_origins,
        allowed_ports=set(settings.allowed_ports),
        exempt_hosts=infrastructure_hosts,
        proxy_endpoints=registered_proxy_endpoints,
        exempt_cidrs=tuple(networks),
    )
    # Best-effort eager resolution so connect-time checks have warm pins; each
    # host is preloaded only within a purpose that actually registers it.
    for purpose, origins in exact_origins.items():
        for origin in origins:
            policy.preload_exempt_host(origin.host, purpose, origin.port)
    return policy
