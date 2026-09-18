"""Local filesystem source controls for server deployments.

In ``authenticated-server`` mode, local path sources ("Local File" / "TXT")
are disabled by default. They require both:

* the ``sources:local`` scope on the authenticated identity, and
* at least one configured ``local_source_roots`` directory.

Every submitted path is canonicalized (``~`` expanded, symlinks resolved) and
must remain inside one of the configured roots. Multipart uploads land in the
system temp directory via tempfile and bypass the root check by design.
"""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from ..exceptions import LocalSourceDenied
from .settings import SCOPE_SOURCES_LOCAL, ServerSettings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .auth import Identity

LOCAL_SOURCE_TYPES = frozenset({"Local File", "TXT"})

PathLike = Union[str, os.PathLike]


def _is_relative_to(path: Path, root: Path) -> bool:
    """Python 3.9-compatible Path.is_relative_to."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_roots(roots: Iterable[PathLike]):
    resolved = []
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise LocalSourceDenied(
                f"Configured local source root does not exist or is not a "
                f"directory: {root_path}"
            )
        resolved.append(root_path)
    return resolved


def resolve_within_roots(path: PathLike, roots: Iterable[PathLike]) -> Path:
    """Canonicalize ``path`` and ensure it is contained in one of ``roots``.

    Symlinks are resolved before the containment check, so a link placed inside
    a root but targeting outside is rejected. Returns the resolved absolute
    path on success; raises LocalSourceDenied otherwise.
    """
    root_list = list(roots)
    if not root_list:
        raise LocalSourceDenied(
            "Local path sources are disabled: no local_source_roots are "
            "configured on the server."
        )
    resolved_roots = _resolved_roots(root_list)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    for root in resolved_roots:
        if _is_relative_to(resolved, root):
            return resolved
    raise LocalSourceDenied(
        f"Local path source {resolved} is outside the configured local source "
        f"root directory/root directories."
    )


def is_within_tempdir(path: PathLike) -> bool:
    """True for paths canonicalized under the system temp directory."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return False
    return _is_relative_to(resolved, temp_root)


def is_local_source_type(source_type: Optional[str]) -> bool:
    return source_type in LOCAL_SOURCE_TYPES


def authorize_local_source(
    path: PathLike,
    *,
    settings: ServerSettings,
    identity: Optional["Any"] = None,
) -> Path:
    """Authorize a local path source for a server request.

    Non-strict modes (compatibility/local-trusted) keep historical behavior and
    only canonicalize the path. Strict mode requires the sources:local scope
    and containment within configured roots.
    """
    if not settings.is_strict:
        return Path(path).expanduser().resolve()
    if identity is None or not identity.has_scope(SCOPE_SOURCES_LOCAL):
        raise LocalSourceDenied(
            "Local path sources require the 'sources:local' scope; the "
            "presented identity is not authorized."
        )
    if not settings.local_source_roots:
        raise LocalSourceDenied(
            "Local path sources are disabled: server.local_source_roots is not "
            "configured."
        )
    return resolve_within_roots(path, settings.local_source_roots)
