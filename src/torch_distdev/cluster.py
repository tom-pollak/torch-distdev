import ast
import contextlib
import inspect
import logging
import os
import socket
import sys
import textwrap
import warnings
from functools import cache
from logging.handlers import QueueHandler, QueueListener

import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.utils
import torch.multiprocessing as mp

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
    """Creates and configures the log listener."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(rank)s]: %(message)s"))
    return QueueListener(log_q, handler)


@contextlib.contextmanager
def _logging_context(log_q):
    """Context manager to start/stop the log listener."""
    listener = _mk_log_listener(log_q)
    listener.start()
    try:
        yield
    finally:
        listener.stop()


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


def _exec_cell(src: str):
    """Run arbitrary source in the worker's global scope and return the result of the last expression."""
    try:
        tree = ast.parse(src)
        if not tree.body:
            return None

        last_node = tree.body[-1]

        if isinstance(last_node, ast.Expr):
            exec_nodes = tree.body[:-1]
            eval_node = ast.Expression(body=last_node.value)
            exec_code = compile(
                ast.Module(body=exec_nodes, type_ignores=[]), "<string>", "exec"
            )
            eval_code = compile(eval_node, "<string>", "eval")
            exec(exec_code, globals())
            return eval(eval_code, globals())
        else:
            exec(src, globals())
            return None
    except Exception as e:
        logging.error(f"Error executing cell:\n{src}\nError: {e}")
        raise


class Cluster:
    def __init__(self, backend: str, nprocs: int):
        self.nprocs = nprocs
        self.backend = backend
        self.port = _free_port()
        self._log_q = mp.get_context("spawn").Queue()
        world = nprocs + 1
        self.ctx: mp.ProcessContext = mp.start_processes(  # type: ignore
            self._worker,
            args=(world, self.port, self._log_q, self.backend),
            nprocs=nprocs,
            start_method="spawn",
            join=False,
        )
        # controller -- final rank
        self._worker(
            world - 1, world, self.port, self._log_q, self.backend, controller=True
        )

    @staticmethod
    def _worker(rank, world, port, log_q, backend, controller=False):
        _set_logger(rank, log_q)
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world - 1),  # not including controller
        )
        dist.init_process_group(backend, rank=rank, world_size=world)
        rpc.init_rpc(
            name="controller" if controller else f"w{rank}",
            rank=rank,
            world_size=world,
            rpc_backend_options=rpc.options.TensorPipeRpcBackendOptions(
                init_method=f"tcp://127.0.0.1:{port}"
            ),
        )
        if not controller:
            rpc.shutdown()
            dist.destroy_process_group()

    @cache
    def _register(self, fn):
        """Send `fn`'s source to every worker once."""
        src = textwrap.dedent(inspect.getsource(fn))
        for i in range(self.nprocs):
            rpc.rpc_sync(f"w{i}", _install_source, args=(src, fn.__name__))

    def launch_cell(self, cell: str):
        """Run the notebook cell on all workers."""
        with _logging_context(self._log_q):
            return tuple(
                rpc.rpc_sync(f"w{i}", _exec_cell, args=(cell,))
                for i in range(self.nprocs)
            )

    def launch(self, fn, *a, **kw):
        """Launch a registered function `fn` on all workers."""
        self._register(fn)
        with _logging_context(self._log_q):
            resp = tuple(
                rpc.rpc_sync(f"w{i}", fn, args=a, kwargs=kw) for i in range(self.nprocs)
            )
        return resp

    def close(self):
        rpc.shutdown()
        dist.destroy_process_group()
        self.ctx.join()
