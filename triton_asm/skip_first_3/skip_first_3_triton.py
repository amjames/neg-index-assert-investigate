
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
#     tl.device_assert((0 <= tmp3) & (tmp3 < 2048), "index out of bounds: 0 <= tmp3 < 2048")
    tmp4 = tl.load(in_ptr1 + (x0 + (16384*tmp3)), None)
    tmp6 = tmp5 + 2048
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
#     tl.device_assert((0 <= tmp8) & (tmp8 < 2048), "index out of bounds: 0 <= tmp8 < 2048")
    tmp9 = tl.load(in_ptr3 + (x0 + (16384*tmp8)), None)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 2048
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
#     tl.device_assert((0 <= tmp14) & (tmp14 < 2048), "index out of bounds: 0 <= tmp14 < 2048")
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
