from pathlib import Path
import triton
import os
import subprocess as sp

NVCC_PATH = Path(os.getenv('CONDA_PREFIX')) / 'bin' / 'nvcc'
NVDISASM_PATH = Path(triton.__file__).parent / 'backends' / 'nvidia' / 'bin' / 'nvdisasm'
ROOT = Path(__file__).parent
CU_SRC = ROOT / 'original.cu'

def run_nvcc(source):
    exe_path = source.parent / source.stem
    sp.run([str(NVCC_PATH), str(source), "-o", str(exe_path)], check=True)
    # generate ptx and cubin with nvcc
    for out in ['ptx', 'cubin']:
        out_path = source.parent / f"{source.stem}.{out}"
        sp.run([str(NVCC_PATH), str(source), f"--{out}", "-o", str(out_path)], check=True)

    # sass
    cubin_path = source.parent / f"{source.stem}.cubin"
    sass = sp.run([str(NVDISASM_PATH), "--life-range-mode", "wide", str(cubin_path)], capture_output=True).stdout.decode('utf-8')
    sass_path = source.parent / f"{source.name}.lifetimes.sass"
    sass_path.write_text(sass)


class MappedAssertsKernelSrc:
    src_lines = CU_SRC.read_text().split("\n")
    assert_indices = [i for i, line in enumerate(src_lines) if line.strip().startswith('assert(')]
    n_assert = len(assert_indices)
    
    @classmethod     
    def write_kernel_skip_first_n_asserts(cls, n_assert_skip):
        assert n_assert_skip <= len(cls.assert_indices)
        skipped_asserts = cls.assert_indices[:n_assert_skip]
        out_lines = []
        for i, l in enumerate(cls.src_lines):
            if i in skipped_asserts:
                out_l = f"//{l}"
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
                out_l = f"// {l}"
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


if __name__ == '__main__':
    kernel_gen = MappedAssertsKernelSrc()
    for name, kernel_src in kernel_gen.kernel_sources.items():
        version_path = (ROOT / name).resolve()
        version_path.mkdir(exist_ok=True, parents=True)
        cu_path = version_path / f"{name}.cu"
        cu_path.write_text(kernel_src)
        run_nvcc(cu_path)


