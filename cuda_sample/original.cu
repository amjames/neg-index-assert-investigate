#include <assert.h>
#include <stdio.h>

__device__ inline int64_t wrap_index(const int64_t index_val,
                                     const int64_t dim_size) {
  const int64_t wrapped = index_val + dim_size;
  return index_val < 0 ? wrapped : index_val;
}
__device__ inline int64_t wrap_index_assert(const int64_t index_val,
                                            const int64_t dim_size) {
  const int64_t wrapped = index_val + dim_size;
  int64_t retval = index_val < 0 ? wrapped : index_val;
  assert((0 <= retval) && (retval < dim_size));
  return retval;
}

__global__ void selectAddKernel(
    const float* vals_0, const int64_t* idx_0, 
    const float* vals_1, const int64_t* idx_1,
    const float* vals_2, const int64_t* idx_2, 
    const float* vals_3, const int64_t* idx_3,
    const float* vals_4, const int64_t* idx_4,
    const int64_t n_val_choice, const int64_t val_length, const int64_t n_index,
    float* out
    ){
  for (int64_t outIndex = blockIdx.x * blockDim.x + threadIdx.x;
       outIndex < n_index; outIndex += gridDim.x * blockDim.x) {
    const int64_t srcIndex0_arg = idx_0[outIndex];
    const int64_t srcIndex1_arg = idx_1[outIndex];
    const int64_t srcIndex2_arg = idx_2[outIndex];
    const int64_t srcIndex3_arg = idx_3[outIndex];
    const int64_t srcIndex4_arg = idx_4[outIndex];

    const int64_t index0 = wrap_index(srcIndex0_arg, n_val_choice);
    const int64_t index1 = wrap_index(srcIndex1_arg, n_val_choice);
    const int64_t index2 = wrap_index(srcIndex2_arg, n_val_choice);
    const int64_t index3 = wrap_index(srcIndex3_arg, n_val_choice);
    const int64_t index4 = wrap_index(srcIndex4_arg, n_val_choice);

    assert((0 <= index0) && (index0 < n_val_choice));
    assert((0 <= index1) && (index1 < n_val_choice));
    assert((0 <= index2) && (index2 < n_val_choice));
    assert((0 <= index3) && (index3 < n_val_choice));
    assert((0 <= index4) && (index4 < n_val_choice));
    for (int i = 0; i < val_length; ++i) {
      float v0i = vals_0[index0 * val_length + i];
      float v1i = vals_1[index1 * val_length + i];
      float v2i = vals_2[index2 * val_length + i];
      float v3i = vals_3[index3 * val_length + i];
      float v4i = vals_4[index4 * val_length + i];
      out[outIndex * val_length + i] = v0i + v1i + v2i + v3i + v4i;
    }
  }
}

const int64_t N_SELECT = 256;
const int64_t N_VAL = 16384;
const int64_t N_VAL_CHOICE = 2048;
const int64_t SIZE_VALS = N_VAL * N_VAL_CHOICE * sizeof(float);
const int64_t SIZE_IDX = N_SELECT * sizeof(int64_t);
const int64_t SIZE_OUT = N_SELECT * N_VAL * sizeof(float);



int main(int argc, char **argv){

  //Host malloc
  float* h_V0 = (float *)malloc(SIZE_VALS);
  float* h_V1 = (float *)malloc(SIZE_VALS);
  float* h_V2 = (float *)malloc(SIZE_VALS);
  float* h_V3 = (float *)malloc(SIZE_VALS);
  float* h_V4 = (float *)malloc(SIZE_VALS);
  int64_t* h_I0 = (int64_t *)malloc(SIZE_IDX);
  int64_t* h_I1 = (int64_t *)malloc(SIZE_IDX);
  int64_t* h_I2 = (int64_t *)malloc(SIZE_IDX);
  int64_t* h_I3 = (int64_t *)malloc(SIZE_IDX);
  int64_t* h_I4 = (int64_t *)malloc(SIZE_IDX);

  //init vals
  for (int j = 0; j < N_VAL_CHOICE; ++j) {
    const int64_t offset = j * N_VAL;
    for (int i = 0; i < N_VAL; ++i) {
      h_V0[offset +i] = i * 1.0;
      h_V1[offset +i] = i * -2.0;
      h_V2[offset +i] = i * 3.0;
      h_V3[offset +i] = i * -4.0;
      h_V4[offset +i] = i * 5.0;
    }
  }
  //init index
  for(int i = 0; i < N_SELECT; ++i){
    h_I0[i] = i;
    h_I1[i] = -i;
    h_I2[i] = i;
    h_I3[i] = -i;
    h_I4[i] = i;
  }

  //cuda malloc
  float* d_V0, *d_V1, *d_V2, *d_V3, *d_V4, *d_OUT;
  int64_t *d_I0, *d_I1, *d_I2, *d_I3, *d_I4;
  cudaMalloc((void **) &d_V0, SIZE_VALS);
  cudaMalloc((void **) &d_V1, SIZE_VALS);
  cudaMalloc((void **) &d_V2, SIZE_VALS);
  cudaMalloc((void **) &d_V3, SIZE_VALS);
  cudaMalloc((void **)&d_V4, SIZE_VALS);
  cudaMalloc((void **) &d_I0, SIZE_IDX);
  cudaMalloc((void **) &d_I1, SIZE_IDX);
  cudaMalloc((void **) &d_I2, SIZE_IDX);
  cudaMalloc((void **) &d_I3, SIZE_IDX);
  cudaMalloc((void **)&d_I4, SIZE_IDX);
  cudaMalloc((void **) &d_OUT, SIZE_OUT);

  //Transfer host -> device

  cudaMemcpy(d_V0, h_V0, SIZE_VALS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V1, h_V1, SIZE_VALS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V2, h_V2, SIZE_VALS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V3, h_V3, SIZE_VALS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V4, h_V4, SIZE_VALS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I0, h_I0, SIZE_IDX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I1, h_I1, SIZE_IDX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I2, h_I2, SIZE_IDX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I3, h_I3, SIZE_IDX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I4, h_I4, SIZE_IDX, cudaMemcpyHostToDevice);

  selectAddKernel<<<128, 256>>>(
      d_V0, d_I0,
      d_V1, d_I1,
      d_V2, d_I2,
      d_V3, d_I3,
      d_V4, d_I4,
      N_VAL_CHOICE, N_VAL, N_SELECT, d_OUT);
  //SYNC
  cudaDeviceSynchronize();

  //Alloc out host
  float* h_OUT = (float *)malloc(SIZE_OUT);
  cudaMemcpy(h_OUT, d_OUT, SIZE_OUT, cudaMemcpyDeviceToHost);
  if(argc > 1) {
    for (int i = 0; i < N_SELECT; ++i) {
      for (int j = 0; j < N_VAL; ++j) {
        int offset = i * N_VAL + j;
        printf("OUT[%d, %d] = %f\n", i, j, h_OUT[offset]);
      }
    }
  }

  cudaFree(d_V0);
  cudaFree(d_V1);
  cudaFree(d_V2);
  cudaFree(d_V3);
  cudaFree(d_V4);
  cudaFree(d_I0);
  cudaFree(d_I1);
  cudaFree(d_I2);
  cudaFree(d_I3);
  cudaFree(d_I4);
  cudaFree(d_OUT);

  free(h_V0);
  free(h_V1);
  free(h_V2);
  free(h_V3);
  free(h_V4);
  free(h_I0);
  free(h_I1);
  free(h_I2);
  free(h_I3);
  free(h_I4);
  free(h_OUT);

  return 0;

}
