import sys
import textwrap

from .singleton import get_cluster

__all__ = ["_maybe_register_magic"]

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
        return cl.launch_cell(cell)

