from .cluster import Cluster
from .magic import _maybe_register_magic  # noqa: F401  registers magic
from .singleton import get_cluster, init_dist

__all__ = ["Cluster", "get_cluster", "init_dist"]
