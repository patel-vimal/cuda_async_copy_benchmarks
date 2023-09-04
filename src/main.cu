#include "kernels/1_thread_block_level_async_copy.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cassert>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void run_kernel_1(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  const size_t nrEltToProcessPerBlock = 1024;
  size_t sharedMemoryRequirement = nrEltToProcessPerBlock * sizeof(float);
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  printf("Launching the kernel...\n");
  sum_simple<nrEltToProcessPerBlock>
      <<<nrBlocks, nrThreadsPerBlock, sharedMemoryRequirement>>>(
          devicePtr, N, deviceOutputPtr);
}

const int KERNEL_COUNT = 1;

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Please select a kernel (range 1 to %d).\n", KERNEL_COUNT-1);
    exit(EXIT_FAILURE);
  }

  // cuda kernel num
  int kernel_num = atoi(argv[1]);
  if (kernel_num < 1 || kernel_num > KERNEL_COUNT) {
    printf("Please enter a valid kernel number (1-%d).\n", KERNEL_COUNT-1);
    exit(EXIT_FAILURE);
  } else {
    printf("Selected kernel %d.\n", kernel_num);
  };

  const size_t N = (size_t(1) << 10);
  assert(N > 0 && "please have meaningful input size");
  float *input = (float *)malloc(N * sizeof(float));
  float ref_output = 0.0;
  for (size_t i = 0; i < N; i++)
    ref_output += (input[i] = float(i));
  float output = 0.0;
  float *devicePtr;
  gpuErrchk(cudaMalloc((void **)&devicePtr, (N + 1) * sizeof(float)));
  float *deviceOutputPtr = &devicePtr[N];
  gpuErrchk(cudaMemcpy(devicePtr, input, N * sizeof(input[0]),
                       cudaMemcpyHostToDevice));

  switch (kernel_num) {
  case 1:
    run_kernel_1(devicePtr, deviceOutputPtr, N);
    break;
  default:
    assert(false && "Found a new bug!");
  }

  // Copy back the output to host.
  gpuErrchk(cudaMemcpy(&output, deviceOutputPtr, sizeof(output),
                       cudaMemcpyDeviceToHost));
  
  // Check for the correctnest of the output.
  printf("ref_output: %f\n", ref_output);
  printf("output: %f\n", output);
  if (output != ref_output) {
    printf("Verification failed!\n");
    exit(-1);
  }
  printf("Verification passed!\n");
  return 0;
}
