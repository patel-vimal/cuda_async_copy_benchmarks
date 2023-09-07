#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

namespace kernel_5 {
const int BLOCK_TILE_SIZE = 128;
const size_t PADDING = 5;
__device__ __noinline__ void
copy_to_shared(float4 sharedTile[BLOCK_TILE_SIZE / 16][4 + PADDING], void *src) {
  float4 *castedSrc = reinterpret_cast<float4 *>(src);
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
  if (i < 32)
    cuda::memcpy_async(thread, &sharedTile[y][x], &castedSrc[i], sizeof(float4),
                       pipeline);
  pipeline.producer_commit();
  pipeline.consumer_wait();
  pipeline.consumer_release();
}

// Experiment:
// Launch config: 1 thread blocks, 128 threads per block, n=128
//
// Shape of the SMEM buffer: 8 x (4+PADDING) x vector<4xf32>.
// Each thread copies a single vector<4xf32> from GMEM to SMEM using cp.async
// that is total of 512B of data per warp.
// ========================================================================
// value of PADDING:            0(0B) | 1(16B) | 4(64B) | 5(80B) | 8(128B)|
// Number of SMEM wavefronts:   8     | 23     |  16    | 23     |   12   |
// Number of GMEM wavefronts:   1     |  2     |   2    |  3     |    2   |
// Number of GMEM sectors read: 16    | 24     |  16    | 24     |   16   |
// ========================================================================
//
// PADDING controls if the SMEM buffer written to by warp using cp.async will be
// contiguous or not. Each threads copies 16B of data using a single cp.async
// instruction. When using PADDING=1 there will be 16B of padding at each of the
// 64B of actual data. That means the SMEM destination locations will be
// contiguous for thread 0-3, 4-7 and so on.
//
// Conclusion: 128B contiguous SMEM chunk gives best performance among all the
// different configurations tried.
// 
// Computes the sum of all the values in the
// array `A` and stores the result in the output. This kernel uses simple
// synchronous copy from global to shared memory.
__global__ void sum_with_thread_wide_memcpy_async(float *input, size_t n, float *output) {
  // Load the data from the global memory to the shared memory to be processed
  // by the current thread block.
  size_t blockStart = blockIdx.x * BLOCK_TILE_SIZE;
  size_t blockEnd = min(blockStart + BLOCK_TILE_SIZE, n);
  assert(n % BLOCK_TILE_SIZE == 0);
  assert(BLOCK_TILE_SIZE == 128);
  // We pad 16B between two rows.
  __shared__ float4 sharedTile2D[BLOCK_TILE_SIZE / 16][4 + PADDING];
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