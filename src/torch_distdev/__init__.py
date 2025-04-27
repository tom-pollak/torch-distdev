from .cluster import Cluster
from .singleton import get_cluster, init_dist
from .magic import _maybe_register_magic  # noqa: F401  registers magic

__all__ = ["Cluster", "get_cluster", "init_dist"]
