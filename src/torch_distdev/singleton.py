from .cluster import Cluster

_GLOBAL_CLUSTER: Cluster | None = None


def init_dist(nprocs: int = 4) -> Cluster:
    """Create (or return) the global cluster."""
    global _GLOBAL_CLUSTER
    if _GLOBAL_CLUSTER is None:
        _GLOBAL_CLUSTER = Cluster(nprocs=nprocs)
    return _GLOBAL_CLUSTER


def get_cluster() -> Cluster | None:
    """Return the cluster if initialised."""
    return _GLOBAL_CLUSTER
