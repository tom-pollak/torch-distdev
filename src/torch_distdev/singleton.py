from .cluster import Cluster

_GLOBAL_CLUSTER: Cluster | None = None


def init_dist(backend: str, nprocs: int) -> Cluster:
    """Create (or return) the global cluster."""
    global _GLOBAL_CLUSTER
    if _GLOBAL_CLUSTER is None:
        _GLOBAL_CLUSTER = Cluster(backend=backend, nprocs=nprocs)
    return _GLOBAL_CLUSTER


def get_cluster() -> Cluster | None:
    """Return the cluster if initialised."""
    return _GLOBAL_CLUSTER
