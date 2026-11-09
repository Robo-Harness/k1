"""Public multi-effector registry without mutating the single-arm runtime."""

from types import SimpleNamespace

from .runtime import ToolRegistry, schema
from .multi_effector_tools import install


_namespace = SimpleNamespace(ToolRegistry=ToolRegistry, schema=schema)
install(_namespace)
MultiEffectorRegistry = _namespace.ToolRegistry
