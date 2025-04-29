from .cluster import Cluster

_GLOBAL_CLUSTER: Cluster | None = None


def init_dist(device, nprocs: int) -> Cluster:
    """Create (or return) the global cluster."""
    global _GLOBAL_CLUSTER
    if _GLOBAL_CLUSTER is not None:
        raise ValueError("Cannot intialize Cluster more than once!")
    _GLOBAL_CLUSTER = Cluster(device=device, nprocs=nprocs)
    return _GLOBAL_CLUSTER

def destroy_dist():
    if cluster := get_cluster() is not None:
        cluster.close()

def get_cluster() -> Cluster | None:
    """Return the cluster if initialised."""
    return _GLOBAL_CLUSTER
