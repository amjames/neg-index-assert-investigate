#Notes

1. `reproducer_torch_source.py`: -> run this script to generate base kernels in the local `inductor_cache` folder.
2. Copy and paste the kernel source code into the `ORIGINAL_SOURCE_TRITON` variable in `run_all_triton.py`
    - If you don't modify the torch source you can just update the `@triton_heruistic.pointwise` args for your system
      the generated code should be the same. 
    - if you change the size/number of args you need to update `get_args`, and the grid size in `call`, and the
      `num_gb` line in the main clause of the `run_all_triton.py` script.
3. After running `run_all_triton.py`,  `stats.csv` will be updated with some simple metrics on throughput /register
   pressure
4. The `triton_asm` dir contains the modified kernel source (comments added) and the ir / assembly for that kernel
5. Run the `run_all` script in the cuda_sample directory to generate something similar for a simple raw cuda example.


