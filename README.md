# torch-distdev

**Distributed PyTorch development without the pain.**

Running `torch.distributed` or RPC code inside a notebook normally means
lots of boiler-plate, multiple terminals, or a full Slurm cluster. **torch-distdev**
condenses that down to:

```python
# %%
from torch_distdev import init_dist
init_dist(nprocs=4)

# %%
%%distributed
import os, logging
logging.info("hello from rank %s", os.environ["RANK"])

```

```
[0]: hello from rank 0
[1]: hello from rank 1
[2]: hello from rank 2
[3]: hello from rank 3
```
