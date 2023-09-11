#include "kernels/1_thread_block_level_simple_copy.cuh"
#include "kernels/2_vectorized_copy.cuh"
#include "kernels/3_block_wide_memcpy_async.cuh"
#include "kernels/4_thread_wide_memcpy_async.cuh"
#include "kernels/5_padded_smem_buffer.cuh"
#include "kernels/6_gmem_alignment_test.cuh"
#include "kernels/async_copy_benchmark.cuh"
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda/pipeline>

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

const size_t nrEltToProcessPerBlock = 128;

void run_kernel_1(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t sharedMemoryRequirement = nrEltToProcessPerBlock * sizeof(float);
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  printf("Launching the kernel...\n");
  sum_simple<nrEltToProcessPerBlock>
      <<<nrBlocks, nrThreadsPerBlock, sharedMemoryRequirement>>>(
          devicePtr, N, deviceOutputPtr);
}

void run_kernel_2(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t sharedMemoryRequirement = nrEltToProcessPerBlock * sizeof(float);
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  printf("Launching the kernel...\n");
  sum_with_vectorized_copy<nrEltToProcessPerBlock>
      <<<nrBlocks, nrThreadsPerBlock, sharedMemoryRequirement>>>(
          devicePtr, N, deviceOutputPtr);
}

void run_kernel_3(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t sharedMemoryRequirement = nrEltToProcessPerBlock * sizeof(float);
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  printf("Launching the kernel...\n");
  kernel_3::sum_with_block_wide_memcpy_async<nrEltToProcessPerBlock>
      <<<nrBlocks, nrThreadsPerBlock, sharedMemoryRequirement>>>(
          devicePtr, N, deviceOutputPtr);
}

void run_kernel_4(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t alignmentBytes = 128;
  size_t sharedMemoryRequirement =
      nrEltToProcessPerBlock * sizeof(float) + (alignmentBytes - 1);
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  printf("Launching the kernel...\n");
  kernel_4::sum_with_thread_wide_memcpy_async<nrEltToProcessPerBlock>
      <<<nrBlocks, nrThreadsPerBlock, sharedMemoryRequirement>>>(
          devicePtr, N, deviceOutputPtr);
}

void run_kernel_5(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  assert(nrThreadsPerBlock == 128);
  assert(nrEltToProcessPerBlock == 128);
  assert(N % 128 == 0);
  printf("Launching the kernel...\n");
  kernel_5::sum_with_thread_wide_memcpy_async
      <<<nrBlocks, nrThreadsPerBlock>>>(devicePtr, N, deviceOutputPtr);
}

void run_kernel_6(float *devicePtr, float *deviceOutputPtr, size_t N) {
  // Compute launch configuration params.
  size_t nrThreadsPerBlock = 128;
  size_t nrBlocks = size_t(std::ceil(float(N) / nrEltToProcessPerBlock));
  assert(nrThreadsPerBlock == 128);
  assert(nrEltToProcessPerBlock == 128);
  assert(N % 128 == 0);
  printf("Launching the kernel...\n");
  kernel_6::sum_with_thread_wide_memcpy_async<<<nrBlocks, nrThreadsPerBlock>>>(
      devicePtr, N, deviceOutputPtr);
}

const int KERNEL_COUNT = 7;
const size_t N = (size_t(1) << 7);

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Please select a kernel (range 1 to %d).\n", KERNEL_COUNT);
    exit(EXIT_FAILURE);
  }

  // cuda kernel num
  int kernel_num = atoi(argv[1]);
  if (kernel_num < 1 || kernel_num > KERNEL_COUNT) {
    printf("Please enter a valid kernel number (1-%d).\n", KERNEL_COUNT);
    exit(EXIT_FAILURE);
  } else {
    printf("Selected kernel %d.\n", kernel_num);
  };

  if (kernel_num == 7) {
    if (argc!=11) {
      printf("Benchmarking requires 10 arguments.\n");
      printf("./main 7 `gmemFirstDimSize` `gmemSecondDimSize` "
             "`gmemConsecutiveRowsPaddingBytes` `gmemAlignmentBytes` "
             "`smemFirstDimSize` `smemSecondDimSize` "
             "`smemConsecutiveRowsPaddingBytes` `smemAlignmentBytes` "
             "`vectorSize`\n");
      return -1;
    }
    size_t gmemFirstDimSize = atoi(argv[2]);
    size_t gmemSecondDimSize = atoi(argv[3]);
    size_t gmemConsecutiveRowsPaddingBytes = atoi(argv[4]);
    size_t gmemAlignmentBytes = atoi(argv[5]);
    size_t smemFirstDimSize = atoi(argv[6]);
    size_t smemSecondDimSize = atoi(argv[7]);
    size_t smemConsecutiveRowsPaddingBytes = atoi(argv[8]);
    size_t smemAlignmentBytes = atoi(argv[9]);
    size_t vectorSize = atoi(argv[10]);
    async_copy_benchmark::asyncCopyBenchmarkRun(
        gmemFirstDimSize, gmemSecondDimSize, gmemConsecutiveRowsPaddingBytes,
        gmemAlignmentBytes, smemFirstDimSize, smemSecondDimSize,
        smemConsecutiveRowsPaddingBytes, smemAlignmentBytes, vectorSize);
    return 0;
  }

  assert(N > 0 && "please have meaningful input size");
  float *input = (float *)malloc(N * sizeof(float));
  float ref_output = 0.0;
  for (size_t i = 0; i < N; i++)
    ref_output += (input[i] = float(i % 5) - 2);
  float output = 0.0;
  float *devicePtr;
  gpuErrchk(cudaMalloc((void **)&devicePtr,
                       ((N + 1 + (kernel_num == 6) * 1024)) * sizeof(float)));
  float *deviceOutputPtr = &devicePtr[N];
  gpuErrchk(cudaMemcpy(devicePtr, input, N * sizeof(input[0]),
                       cudaMemcpyHostToDevice));

  switch (kernel_num) {
  case 1:
    run_kernel_1(devicePtr, deviceOutputPtr, N);
    break;
  case 2:
    run_kernel_2(devicePtr, deviceOutputPtr, N);
    break;
  case 3:
    run_kernel_3(devicePtr, deviceOutputPtr, N);
    break;
  case 4:
    run_kernel_4(devicePtr, deviceOutputPtr, N);
    break;
  case 5:
    run_kernel_5(devicePtr, deviceOutputPtr, N);
    break;
  case 6:
    run_kernel_6(devicePtr, deviceOutputPtr, N);
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
    printf("\033[1;31mVerification failed!\033[0m\n");
    exit(-1);
  }
  printf("\033[1;32mVerification passed!\033[0m\n");
  return 0;
}
