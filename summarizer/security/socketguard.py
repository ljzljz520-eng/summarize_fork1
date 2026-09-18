"""Process-wide socket shims enforcing the current outbound policy.

The guard patches ``socket.getaddrinfo``/``gethostbyname`` (so EVERY resolved
address is checked, not just the first one) and ``socket.socket.connect``
(so the actual connection target is checked at connect time, defeating DNS
rebinding). It is idempotent and reference-counted; the installed hooks read
the current :class:`OutboundPolicy` from a context variable and are no-ops for
the permissive policy used by the standalone CLI.
"""

import errno
import functools
import socket
import threading
from contextlib import contextmanager
from typing import Any, Tuple

from .netpolicy import OutboundPolicyError, get_policy

_LOCK = threading.RLock()
_INSTALLED = False
_DEPTH = 0
# Process-wide strict fallback. The per-request ContextVar policy wins; the
# fallback closes the gap where third-party libraries perform network I/O in
# self-created worker threads that do not inherit the request ContextVar.
_FALLBACK_POLICY = None

_ORIGINALS: dict = {}


def _effective_policy():
    policy = get_policy()
    if policy.enforce:
        return policy
    if _FALLBACK_POLICY is not None and _FALLBACK_POLICY.enforce:
        return _FALLBACK_POLICY
    return policy


def _guarded_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    results = _ORIGINALS["getaddrinfo"](host, port, family, type, proto, flags)
    policy = _effective_policy()
    if policy.enforce:
        checked = set()
        for entry in results:
            sockaddr = entry[4]
            ip = sockaddr[0]
            if ip in checked:
                continue
            checked.add(ip)
            # IPv6 flow/scope suffix stripped by sockaddr already.
            policy.check_endpoint_ip(ip, host=host, port=port)
    return results


def _guarded_gethostbyname(host):
    ip = _ORIGINALS["gethostbyname"](host)
    policy = _effective_policy()
    if policy.enforce:
        policy.check_endpoint_ip(ip, host=host)
    return ip


def _guarded_gethostbyname_ex(host):
    result = _ORIGINALS["gethostbyname_ex"](host)
    policy = _effective_policy()
    if policy.enforce:
        _hostname, _aliases, ip_list = result
        for ip in ip_list:
            policy.check_endpoint_ip(ip, host=host)
    return result


def _validate_connect_address(sock: Any, address: Any) -> None:
    """Validate the target of socket.connect/connect_ex.

    Addresses are normally already-resolved (ip, port) tuples; some callers
    pass unresolved hostnames, which are resolved via the ORIGINAL resolver
    and then fully validated. Non-IP sockets (AF_UNIX etc.) are untouched.
    """
    policy = _effective_policy()
    if not policy.enforce:
        return
    family = getattr(sock, "family", None)
    if family not in (socket.AF_INET, socket.AF_INET6):
        return
    if isinstance(address, tuple) and address:
        host = address[0]
        port = address[1] if len(address) > 1 else None
    else:
        host = address
        port = None
    host = host.strip("[]") if isinstance(host, str) else host
    try:
        # Pass the value both as target and origin host: IP-literal targets
        # can never match a hostname-scoped pin.
        policy.check_connect_ip(host, host=host, port=port)
        return
    except ValueError:
        # Not an IP literal: resolve via the original resolver and validate
        # every returned address (registered hostnames pin within the active
        # purpose resolution scope).
        pass
    results = _ORIGINALS["getaddrinfo"](host, port or 0)
    seen = set()
    for entry in results:
        ip = entry[4][0]
        if ip in seen:
            continue
        seen.add(ip)
        policy.check_endpoint_ip(ip, host=host, port=port)


def _guarded_connect(self, address):
    _validate_connect_address(self, address)
    return _ORIGINALS["connect"](self, address)


def _guarded_connect_ex(self, address):
    try:
        _validate_connect_address(self, address)
    except OutboundPolicyError:
        # connect_ex reports failures as errno-style return codes.
        return errno.ECONNREFUSED
    return _ORIGINALS["connect_ex"](self, address)


def install_socket_guard(fallback_policy=None) -> int:
    """Install the socket shims once; nested installs are reference counted.

    ``fallback_policy`` is the strict policy enforced in threads that do not
    inherit the per-request policy ContextVar (third-party worker threads).
    """
    global _INSTALLED, _DEPTH, _FALLBACK_POLICY
    with _LOCK:
        if fallback_policy is not None and fallback_policy.enforce:
            _FALLBACK_POLICY = fallback_policy
        if not _INSTALLED:
            _ORIGINALS["getaddrinfo"] = socket.getaddrinfo
            _ORIGINALS["gethostbyname"] = socket.gethostbyname
            _ORIGINALS["gethostbyname_ex"] = socket.gethostbyname_ex
            _ORIGINALS["connect"] = socket.socket.connect
            _ORIGINALS["connect_ex"] = socket.socket.connect_ex
            socket.getaddrinfo = _guarded_getaddrinfo
            socket.gethostbyname = _guarded_gethostbyname
            socket.gethostbyname_ex = _guarded_gethostbyname_ex
            socket.socket.connect = _guarded_connect
            socket.socket.connect_ex = _guarded_connect_ex
            _INSTALLED = True
        _DEPTH += 1
        return _DEPTH


def uninstall_socket_guard() -> int:
    """Remove one install reference; restores originals at zero."""
    global _INSTALLED, _DEPTH, _FALLBACK_POLICY
    with _LOCK:
        if _DEPTH <= 0:
            return 0
        _DEPTH -= 1
        if _DEPTH == 0 and _INSTALLED:
            socket.getaddrinfo = _ORIGINALS["getaddrinfo"]
            socket.gethostbyname = _ORIGINALS["gethostbyname"]
            socket.gethostbyname_ex = _ORIGINALS["gethostbyname_ex"]
            socket.socket.connect = _ORIGINALS["connect"]
            socket.socket.connect_ex = _ORIGINALS["connect_ex"]
            _INSTALLED = False
            _FALLBACK_POLICY = None
        return _DEPTH


def socket_guard_active() -> bool:
    return _INSTALLED


@contextmanager
def socket_guard():
    install_socket_guard()
    try:
        yield
    finally:
        uninstall_socket_guard()
