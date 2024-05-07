
# AOT ID: ['0_inference']
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


# kernel path: /home/paperspace/git/neg-index-assert-investigate/inductor_cache/jq/cjq5w7snqyjnlnwjieqrvs25wh3tmce26bfoz7korwq52xf2ezxm.py
# Source Nodes: [add, add_1, add_2, add_3, s_0, s_1, s_2, s_3, s_4], Original ATen: [aten.add, aten.index]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# s_0 => index
# s_1 => index_1
# s_2 => index_2
# s_3 => index_3
# s_4 => index_4
triton_poi_fused_add_index_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6527e0f035aedddc89d3c2e03c61ab7680817bc9f7cff67ebe7cac03c798e486', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 2.819883008},
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


def get_args():
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


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 536870912, grid=grid(536870912), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 536870912, grid=grid(536870912))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 2.819883008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 16384), (16384, 1))
    assert_size_stride(arg1_1, (128, 256), (256, 1))
    assert_size_stride(arg2_1, (2048, 16384), (16384, 1))
    assert_size_stride(arg3_1, (128, 256), (256, 1))
    assert_size_stride(arg4_1, (2048, 16384), (16384, 1))
    assert_size_stride(arg5_1, (128, 256), (256, 1))
    assert_size_stride(arg6_1, (2048, 16384), (16384, 1))
    assert_size_stride(arg7_1, (128, 256), (256, 1))
    assert_size_stride(arg8_1, (2048, 16384), (16384, 1))
    assert_size_stride(arg9_1, (128, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 256, 16384), (4194304, 16384, 1), torch.float32)
        # Source Nodes: [add, add_1, add_2, add_3, s_0, s_1, s_2, s_3, s_4], Original ATen: [aten.add, aten.index]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_0.run(arg1_1, arg0_1, arg3_1, arg2_1, arg5_1, arg4_1, arg7_1, arg6_1, arg9_1, arg8_1, buf0, 536870912, grid=grid(536870912), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg8_1 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
