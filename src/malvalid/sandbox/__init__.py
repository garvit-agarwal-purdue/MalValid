"""Isolated execution of the submitted (untrusted) model.

See :mod:`malvalid.sandbox.host` for the policy, backends and the model proxy,
:mod:`malvalid.sandbox.worker` for the worker process and its pickle guards, and
:mod:`malvalid.sandbox.protocol` for the no-pickle wire format.
"""

from malvalid.sandbox.host import (
    BACKENDS,
    ISOLATION_LEVELS,
    InProcessModel,
    SandboxedModel,
    SandboxPolicy,
    inspect_adapter,
    isolation_level,
    open_model,
    os_sandbox_available,
    probe_backends,
    select_backend,
)

__all__ = [
    "BACKENDS",
    "ISOLATION_LEVELS",
    "InProcessModel",
    "SandboxPolicy",
    "SandboxedModel",
    "inspect_adapter",
    "isolation_level",
    "open_model",
    "os_sandbox_available",
    "probe_backends",
    "select_backend",
]
