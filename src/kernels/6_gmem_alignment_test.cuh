#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

namespace kernel_6 {
const int BLOCK_TILE_SIZE = 128;
const size_t GMEM_OFFSET = 8;
__device__ __noinline__ void
copy_to_shared(float4 sharedTile[BLOCK_TILE_SIZE / 16][4], void *src) {
  float4 *castedSrc = reinterpret_cast<float4 *>(src);
  castedSrc += GMEM_OFFSET;
  auto block = cooperative_groups::this_thread_block();
  auto thread = cooperative_groups::this_thread();
  // Create a pipeline.
  constexpr size_t stages_count = 1;
  auto pipeline = cuda::make_pipeline();
  size_t i = threadIdx.x;
  // Compute the src and dst buffer addresses of the element we are going to
  // process.
  size_t x = i % 4;
  size_t y = i / 4;
  pipeline.producer_acquire();
  if (i < 32)
    cuda::memcpy_async(thread, &sharedTile[y][x], &castedSrc[i], sizeof(float4),
                       pipeline);
  pipeline.producer_commit();
  pipeline.consumer_wait();
  pipeline.consumer_release();
}
//
// Alignment value(Bytes):       |  0 | 16 | 32 | 128 | 512
// Number of GMEM sectors read:  | 16 | 32 | 16 |  16 |  16
// Number of SMEM wavefronts:    |  8 | 16 | 12 |   8 |   8
// Conclusion: We should use at-least 32B of alignment to ensure that the number
// of extra GMEM sectors remain optimal.
__global__ void sum_with_thread_wide_memcpy_async(float *input, size_t n, float *output) {
  // Load the data from the global memory to the shared memory to be processed
  // by the current thread block.
  size_t blockStart = blockIdx.x * BLOCK_TILE_SIZE;
  size_t blockEnd = min(blockStart + BLOCK_TILE_SIZE, n);
  assert(n % BLOCK_TILE_SIZE == 0);
  assert(BLOCK_TILE_SIZE == 128);
  // We pad 16B between two rows.
  __shared__ float4 sharedTile2D[BLOCK_TILE_SIZE / 16][4];
  copy_to_shared(sharedTile2D, &input[blockStart]);

  // Wait for all the copies to finish.
  __syncthreads();
  __shared__ float sharedTile[BLOCK_TILE_SIZE];
  if (threadIdx.x < 32) {
    *((float4 *)&sharedTile[threadIdx.x * 4]) =
        sharedTile2D[threadIdx.x / 4][threadIdx.x % 4];
  }
  __syncthreads();
  // Compute the sum of all the elements in the tile.
  float perThreadSum = 0.0;
  for (size_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x)
    perThreadSum += sharedTile[i - blockStart];
  __syncthreads();
  // Store the sum in the allignedSharedTile[threadId.x];
  sharedTile[threadIdx.x] = perThreadSum;
  // Have the first thread compute the sum of all the elements in the
  // allignedSharedTile.
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  perThreadSum = 0.0;
  for (size_t i = 0; i < blockDim.x; i++)
    perThreadSum += sharedTile[i];
  // Atomically add the result to the `output`.
  atomicAdd(output, perThreadSum);
  return;
}

} // namespace kernel_3