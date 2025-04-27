import sys
import os
import socket
import inspect
import textwrap
import logging
from logging.handlers import QueueListener, QueueHandler
import warnings
from functools import cache

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc

warnings.filterwarnings(
    "ignore",
    message=r"You are using a Backend .*",
    category=UserWarning,
)


def _install_source(src: str, name: str):
    """Executed on each worker - execs `src` into globals()."""
    ns = {}
    exec(src, ns)
    sys.modules["__main__"].__dict__[name] = ns[name]


def _mk_log_listener(log_q):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(rank)s]: %(message)s"))
    return QueueListener(log_q, handler)


def _set_logger(rank, log_q):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    qh = QueueHandler(log_q)
    qh.addFilter(lambda rec, r=rank: setattr(rec, "rank", r) or True)
    root.addHandler(qh)


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return str(p)


class Cluster:
    def __init__(self, nprocs=4):
        self.nprocs = nprocs
        self.port = _free_port()
        self._log_q = mp.get_context("spawn").Queue()
        world = nprocs + 1
        self.ctx = mp.start_processes(
            self._worker,
            args=(world, self.port, self._log_q),
            nprocs=nprocs,
            start_method="spawn",
            join=False,
        )
        # controller -- final rank
        self._worker(world - 1, world, self.port, self._log_q, controller=True)

    @staticmethod
    def _worker(rank, world, port, log_q, controller=False):
        _set_logger(rank, log_q)
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world)
        rpc.init_rpc(
            name="controller" if controller else f"w{rank}",
            rank=rank,
            world_size=world,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://127.0.0.1:{port}"
            ),
        )
        if not controller:
            rpc.shutdown()
            dist.destroy_process_group()

    @cache
    def _register(self, fn):
        """Send `fn`'s source to every worker once."""
        if hasattr(fn, "__source__"):
            src = fn.__source__
        else:
            src = textwrap.dedent(inspect.getsource(fn))
        for i in range(self.nprocs):
            rpc.rpc_sync(f"w{i}", _install_source, args=(src, fn.__name__))

    def launch(self, fn, *a, **kw):
        listener = _mk_log_listener(self._log_q)
        listener.start()
        self._register(fn)
        try:
            resp = tuple(
                [
                    rpc.rpc_sync(f"w{i}", fn, args=a, kwargs=kw)
                    for i in range(self.nprocs)
                ]
            )
        finally:
            listener.stop()
        return resp

    def close(self):
        rpc.shutdown()
        dist.destroy_process_group()
        self.ctx.join()
