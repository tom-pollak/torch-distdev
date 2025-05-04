from .cluster import Cluster
from .singleton import get_cluster, init_dist, destroy_dist
from .magic import _maybe_register_magic


__all__ = ["Cluster", "init_dist", "destroy_dist", "get_cluster"]

_maybe_register_magic()
