"""Policy-aware HTTP client factories for requests and aiohttp.

Every outbound HTTP call in the application must go through one of these:

* :class:`GuardedSession` / :func:`session_for` for blocking ``requests``
  clients (Drive, Dropbox, Cobalt);
* :func:`guarded_aiohttp_session` / :func:`aiohttp_session_for` for async
  callers (model/vision chat completions);
* :func:`preflight_url` for third-party libraries that own their client
  (yt-dlp, youtube-transcript-api, litellm) — the socket guard provides the
  IP/redirect backstop for those.
"""

from typing import Optional

from .netpolicy import (
    EXACT_ORIGIN_PURPOSES,
    OutboundPolicy,
    get_policy,
    normalize_origin,
    resolution_scope,
)


def preflight_url(policy: Optional[OutboundPolicy], url: str, purpose: str):
    """URL-layer check for libraries that build their own HTTP clients.

    Returns the normalized :class:`~netpolicy.Origin` of the checked URL.
    """
    return (policy or get_policy()).check_url(url, purpose)


# ── requests ────────────────────────────────────────────────────────────────


def _build_requests():
    import requests
    from requests.adapters import HTTPAdapter

    class _GuardedAdapter(HTTPAdapter):
        """Validates every request, including each redirect hop.

        requests re-enters ``adapter.send`` for every resolved redirect, so a
        single check here covers the whole redirect chain.
        """

        def __init__(self, policy, purpose, **kwargs):
            self._policy = policy
            self._purpose = purpose
            super().__init__(**kwargs)

        def send(self, request, *args, **kwargs):  # type: ignore[override]
            self._policy.check_url(request.url, self._purpose)
            # Bind the resolution/connect hooks to this purpose for the whole
            # (synchronous) urllib3 call, including each redirect re-entry.
            with resolution_scope(self._purpose):
                return super().send(request, *args, **kwargs)

    class _GuardedSession(requests.Session):
        def __init__(
            self,
            policy,
            purpose,
            *,
            proxies=None,
            trust_env=None,
            max_redirects=None,
        ):
            super().__init__()
            self._policy = policy
            self._purpose = purpose
            self._forbid_redirects = bool(
                policy.enforce and purpose in EXACT_ORIGIN_PURPOSES
            )
            adapter = _GuardedAdapter(policy, purpose)
            self.mount("http://", adapter)
            self.mount("https://", adapter)
            # Never inherit ambient proxy settings: proxies must be the
            # server-registered ones passed explicitly.
            self.trust_env = (
                (not policy.enforce) if trust_env is None else trust_env
            )
            if proxies:
                self.proxies.update(proxies)
            if max_redirects is not None and not self._forbid_redirects:
                # requests reads this attribute in resolve_redirects.
                self.max_redirects = max_redirects

        def request(self, method, url, **kwargs):  # type: ignore[override]
            if self._forbid_redirects:
                kwargs["allow_redirects"] = False
            return super().request(method, url, **kwargs)

        def should_strip_auth(self, old_url, new_url):
            # Never leak credentials across origins; cover same-host/different
            # port cases that the default implementation misses.
            try:
                if normalize_origin(old_url) != normalize_origin(new_url):
                    return True
            except Exception:
                return True
            return super().should_strip_auth(old_url, new_url)

    return _GuardedSession, _GuardedAdapter


def session_for(
    policy: Optional[OutboundPolicy],
    purpose: str,
    *,
    proxies=None,
    trust_env=None,
    **kwargs,
):
    """Create a GuardedSession bound to (policy, purpose)."""
    session_cls, _adapter_cls = _build_requests()
    return session_cls(
        policy or get_policy(),
        purpose,
        proxies=proxies,
        trust_env=trust_env,
        **kwargs,
    )


# ── aiohttp ─────────────────────────────────────────────────────────────────


class GuardedAiohttpSession:
    """Composition wrapper forcing allow_redirects=False for bound origins."""

    def __init__(self, session, *, forbid_redirects=False, purpose=None):
        self._session = session
        self.guarded_forbid_redirects = forbid_redirects
        self._purpose = purpose

    async def request(self, method, url, **kwargs):
        if self.guarded_forbid_redirects:
            kwargs["allow_redirects"] = False
        # The connect() backstop runs in this same event-loop task; keep the
        # purpose scope active for the whole request including redirects.
        with resolution_scope(self._purpose):
            return await self._session.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)

    async def close(self):
        await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __getattr__(self, item):
        return getattr(self._session, item)


def _build_aiohttp_connector():
    import aiohttp

    class _GuardedTCPConnector(aiohttp.TCPConnector):
        """Validates every resolved address in the event loop thread.

        Resolution is validated here (rather than relying solely on the socket
        shims) because aiohttp resolves names in an executor thread that does
        not inherit the request ContextVar; validating here also pins exempt
        infrastructure IPs before ``connect`` runs.
        """

        def __init__(self, *args, policy=None, purpose=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._bound_policy = policy
            self._bound_purpose = purpose

        async def _resolve_host(self, *args, **kwargs):
            # aiohttp runs this method in an executor thread that does not
            # inherit the request ContextVar: install both the fallback policy
            # (via the socket shims) and the purpose scope locally.
            with resolution_scope(self._bound_purpose):
                entries = await super()._resolve_host(*args, **kwargs)
                policy = self._bound_policy or get_policy()
                if not policy.enforce:
                    return entries
                host = args[0] if args else kwargs.get("host")
                port = args[1] if len(args) > 1 else kwargs.get("port")
                for entry in entries:
                    ip = entry.get("host") or entry.get("ip")
                    if ip is not None:
                        policy.check_endpoint_ip(
                            ip,
                            host=host,
                            port=port,
                            purpose=self._bound_purpose,
                        )
                return entries

    return _GuardedTCPConnector


def guarded_aiohttp_session(
    policy: Optional[OutboundPolicy] = None,
    purpose: Optional[str] = None,
    *,
    allow_redirects: Optional[bool] = None,
    connector_kwargs=None,
    trust_env: bool = False,
    **session_kwargs,
):
    """Create an aiohttp ClientSession with a policy-guarded connector.

    Exact-origin purposes (model/vision/cobalt_api) disable redirects by
    default. Callers must run :meth:`OutboundPolicy.check_url` (or
    :func:`preflight_url`) on the request URL before sending.
    """
    import aiohttp

    bound_policy = policy or get_policy()
    connector_cls = _build_aiohttp_connector()
    connector = connector_cls(
        policy=bound_policy,
        purpose=purpose,
        **(connector_kwargs or {}),
    )
    kwargs = {"trust_env": trust_env}
    kwargs.update(session_kwargs)
    if allow_redirects is None:
        forbid = bound_policy.enforce and purpose in EXACT_ORIGIN_PURPOSES
    else:
        forbid = not allow_redirects
    session = aiohttp.ClientSession(connector=connector, **kwargs)
    return GuardedAiohttpSession(
        session, forbid_redirects=forbid, purpose=purpose
    )


# Friendly alias kept for call-site readability.
aiohttp_session_for = guarded_aiohttp_session


# ── module-level function wrapper (permissive path keeps `requests.get`) ────


def guarded_request(
    method: str,
    url: str,
    *,
    policy: Optional[OutboundPolicy] = None,
    purpose: Optional[str] = None,
    **kwargs,
):
    """Call ``requests.<method>`` through the outbound policy.

    Permissive policies (standalone CLI) use the plain module-level
    ``requests`` function, preserving historical behavior exactly. Strict
    policies use a GuardedSession so every redirect hop is validated.
    """
    import requests

    bound_policy = policy or get_policy()
    bound_policy.check_url(url, purpose)
    if bound_policy.enforce:
        session_cls, _adapter_cls = _build_requests()
        session = session_cls(bound_policy, purpose)
        try:
            return session.request(method, url, **kwargs)
        finally:
            session.close()
    return getattr(requests, method.lower())(url, **kwargs)
