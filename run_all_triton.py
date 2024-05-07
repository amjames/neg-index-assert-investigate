from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


ORIGINAL_TRITON_SRC = '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[536870912], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=30), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'ce52ce89546c8ffd8a9cfa4405558e5c797ed169df3771129471a6e0b4baf65e', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 2.819883008},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16384)
    x0 = xindex % 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 2048
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 2048), "index out of bounds: 0 <= tmp3 < 2048")
    tmp4 = tl.load(in_ptr1 + (x0 + (16384*tmp3)), None)
    tmp6 = tmp5 + 2048
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert((0 <= tmp8) & (tmp8 < 2048), "index out of bounds: 0 <= tmp8 < 2048")
    tmp9 = tl.load(in_ptr3 + (x0 + (16384*tmp8)), None)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 2048
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert((0 <= tmp14) & (tmp14 < 2048), "index out of bounds: 0 <= tmp14 < 2048")
    tmp15 = tl.load(in_ptr5 + (x0 + (16384*tmp14)), None)
    tmp16 = tmp10 + tmp15
    tmp18 = tmp17 + 2048
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 2048), "index out of bounds: 0 <= tmp20 < 2048")
    tmp21 = tl.load(in_ptr7 + (x0 + (16384*tmp20)), None)
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23 + 2048
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 2048), "index out of bounds: 0 <= tmp26 < 2048")
    tmp27 = tl.load(in_ptr9 + (x0 + (16384*tmp26)), None)
    tmp28 = tmp22 + tmp27
    tl.store(out_ptr0 + (x2), tmp28, None)
'''

class MappedAssertsKernelSrc:
    src_lines = ORIGINAL_TRITON_SRC.split("\n")
    assert_indices = [i for i, line in enumerate(src_lines) if line.strip().startswith('tl.device_assert(')]
    n_assert = len(assert_indices)
    
    @classmethod     
    def write_kernel_skip_first_n_asserts(cls, n_assert_skip):
        assert n_assert_skip <= len(cls.assert_indices)
        skipped_asserts = cls.assert_indices[:n_assert_skip]
        out_lines = []
        for i, l in enumerate(cls.src_lines):
            if i in skipped_asserts:
                out_l = f"# {l}"
            else:
                out_l = l
            out_lines.append(out_l)
        return "\n".join(out_lines)

    @classmethod 
    def write_kernel_skip_last_asserts(cls, n_assert_skip):
        assert n_assert_skip <= len(cls.assert_indices)
        skipped_asserts = list(reversed(cls.assert_indices))[:n_assert_skip]
        out_lines = []
        for i, l in enumerate(cls.src_lines):
            if i in skipped_asserts:
                out_l = f"# {l}"
            else:
                out_l = l
            out_lines.append(out_l)
        return "\n".join(out_lines)

    def __init__(self):
        self.kernel_sources = {}
        cls = type(self)
        self.kernel_sources['no_skips'] = cls.write_kernel_skip_first_n_asserts(0)
        for i in range(1, cls.n_assert):
            skip_first_src = cls.write_kernel_skip_first_n_asserts(i)
            self.kernel_sources[f'skip_first_{i}'] = skip_first_src
        for i in range(1, cls.n_assert):
            skip_last_src = cls.write_kernel_skip_last_asserts(i)
            self.kernel_sources[f'skip_last_{i}']  = skip_last_src
        self.kernel_sources['skip_all'] = cls.write_kernel_skip_first_n_asserts(cls.n_assert)



def get_args():
    from torch._dynamo.testing import rand_strided
    arg_0 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg_5 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg_7 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg_9 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((128, 256, 16384), (4194304, 16384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10,


def call(kernel, args):
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime.triton_heuristics import grid
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        return kernel.run(*args, 536870912, grid=grid(536870912), stream=stream0)
    

def dump_triton(name, handle, mapped_kernels):
    from pathlib import Path
    import triton
    import subprocess as sp
    triton_dump_dir = Path(__file__).parent / 'triton_asm' / name
    triton_dump_dir.mkdir(exist_ok=True, parents=True)
    paths = {}
    for key, val in handle.asm.items():
        open_flags = 'w' if key != 'cubin' else 'wb'
        dump_f = triton_dump_dir / f"{name}.{key}"
        paths[key] = str(dump_f.resolve())
        with dump_f.open(open_flags) as f:
            f.write(val)
    
    source_file = triton_dump_dir / f"{name}_triton.py"
    source_file.write_text(mapped_kernels.kernel_sources[name])
    paths['triton'] = str(source_file.resolve())
    NVDISASM = Path(triton.__file__).parent / 'backends' / 'nvidia' / 'bin' / 'nvdisasm'
    proc = sp.run([str(NVDISASM), "--life-range-mode", 'wide', paths['cubin']], capture_output=True)
    sass_file = triton_dump_dir / f"{name}.lifetimes.sass"
    sass_file.write_text(proc.stdout.decode("utf-8"))


def write_results_csv(results):
    from pathlib import Path
    import csv
    r_path = Path(__file__).parent / 'stats.csv'
    r0 = results[0]
    fieldnames = list(r0.keys())
    with r_path.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def print_result(result):
    name = result.get('name')
    ms = result.get('ms')
    gb = result.get('num_gb')
    gb_per_s = result.get('gb_per_s')
    n_regs = result.get('n_regs')
    print(f"[{name: <30}] {ms:10.3f}ms {gb:10.3f}GB {gb_per_s:10.3f}GB/s {n_regs} regs")
    

mapped_kernels = MappedAssertsKernelSrc()
kernels_ns = {}
for name, src in mapped_kernels.kernel_sources.items():
    kernels_ns[name] = async_compile.triton('triton_', src, device_str='cuda')

async_compile.wait(kernels_ns)
del async_compile



if __name__ == '__main__':
    from triton.testing import do_bench
    results = []
    args = get_args()
    for name, kernel in kernels_ns.items():
        handle = call(kernel, args)
        n_regs = handle.n_regs
        ms = do_bench(lambda: call(kernel, args), rep=40, fast_flush=True)
        num_gb = 2.819883008
        gb_per_s = num_gb / (ms / 1e3)
        result = {
            'name': name,
            'ms': ms,
            'num_gb': num_gb,
            'gb_per_s': gb_per_s,
            'n_regs': n_regs,
        }
        print_result(result)
        results.append(result)
        dump_triton(name, handle, mapped_kernels),
    write_results_csv(results)



