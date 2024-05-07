import torch
import os
from pathlib import Path

# logging
torch._logging.set_logs(
    output_code=True,
)

# dump kernels to local cache
THIS_DIR = Path(__file__).parent
LOCAL_CACHE = THIS_DIR / 'inductor_cache'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(LOCAL_CACHE.resolve())
os.environ['TORCHINDUCTOR_BENCHMARK_KERNEL'] = "1"


# constants (slightly sized down from https://github.com/pytorch/pytorch/issues/120452)
N_BATCH = 128
N_VAL_CHOICE = 2048
VAL_LENGTH = 16384
N_SELECT = 256

def make_arg_pair():
    vals = torch.randn(N_VAL_CHOICE, VAL_LENGTH, dtype=torch.float32, device='cuda')
    index = torch.zeros(N_BATCH, N_SELECT, dtype=torch.int64, device='cuda')
    return vals, index


def make_args(n_pair=5):
    arg_ps = [make_arg_pair() for _ in range(n_pair)]
    return sum(arg_ps, ())
@torch.compile
def select_and_sum(
        val_0,
        idx_0,
        val_1,
        idx_1,
        val_2,
        idx_2,
        val_3,
        idx_3,
        val_4,
        idx_4,
):
    s_0 = val_0[idx_0]
    s_1 = val_1[idx_1]
    s_2 = val_2[idx_2]
    s_3 = val_3[idx_3]
    s_4 = val_4[idx_4]

    return s_0 + s_1 + s_2 + s_3 + s_4

args = make_args()
result = select_and_sum(*args)
print(result)

