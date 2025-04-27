import sys
import textwrap
from .singleton import get_cluster


def _maybe_register_magic():
    """Register %%distributed if we are running inside IPython/Jupyter."""
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        if ip is None:  # not inside an IPython shell
            return
        from IPython.core.magic import register_cell_magic
    except ImportError:
        return  # IPython not installed - skip

    @register_cell_magic("distributed")
    def _distributed_magic(line, cell):
        """Run the cell on all workers (requires `init_dist`)."""
        cl = get_cluster()
        if cl is None:
            raise RuntimeError("Call `init_dist()` before using %%distributed")

        src = textwrap.indent(cell, "    ")
        code = f"def _dist_user_fn():\n{src}"
        ns = {}
        exec(code, ns)
        fn = ns["_dist_user_fn"]
        fn.__source__ = cell
        return cl.launch(fn)


_maybe_register_magic()
