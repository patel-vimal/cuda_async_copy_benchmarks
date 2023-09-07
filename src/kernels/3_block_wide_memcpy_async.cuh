#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

namespace kernel_3 {
template<typename T>
__device__ __noinline__ void copy_to_shared(void *sharedTile, void *src, size_t nrElementsToCopy) {
  T* castedSharedTile = reinterpret_cast<T*>(sharedTile);
  T *castedSrc = reinterpret_cast<T *>(src);
  auto block = cooperative_groups::this_thread_block();
  // Create a pipeline.
  constexpr size_t stages_count = 1;
  // Allocate shared storage for a single stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,
                                         stages_count>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);
  pipeline.producer_acquire();
  for (size_t i = 0; i < nrElementsToCopy; i += block.size()) {
    size_t remainingElt = nrElementsToCopy - i;
    size_t eltToCopy =
        (block.size() < remainingElt ? block.size() : remainingElt);
    cuda::memcpy_async(block, &castedSharedTile[i], &castedSrc[i],
                       sizeof(T) * eltToCopy, pipeline);
  }
  pipeline.producer_commit();
  pipeline.consumer_wait();
  pipeline.consumer_release();
}

// Computes the sum of all the values in the array `A` and stores the result in
// the output. This kernel uses simple synchronous copy from global to shared
// memory.
template <const int BLOCK_TILE_SIZE>
__global__ void sum_with_block_wide_memcpy_async(float *input, size_t n, float *output) {
  extern __shared__ __align__(128) float sharedTile[];

  // Load the data from the global memory to the shared memory to be processed
  // by the current thread block.
  size_t blockStart = blockIdx.x * BLOCK_TILE_SIZE;
  size_t blockEnd = min(blockStart + BLOCK_TILE_SIZE, n);
  size_t nrElementsToCopy = blockEnd - blockStart;
  // Selects the largest possible vector-width for the copy.
  if (nrElementsToCopy % 4 == 0)
    copy_to_shared<float4>(sharedTile, &input[blockStart],
                           nrElementsToCopy / 4);
  else if (nrElementsToCopy % 2 == 0)
    copy_to_shared<float2>(sharedTile, &input[blockStart],
                           nrElementsToCopy / 2);
  else
    copy_to_shared<float>(sharedTile, &input[blockStart], nrElementsToCopy);

  // Wait for all the copies to finish.
  __syncthreads();
  // Compute the sum of all the elements in the tile.
  float perThreadSum = 0.0;
  for (size_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x)
    perThreadSum += sharedTile[i - blockStart];
  __syncthreads();
  // Store the sum in the sharedTile[threadId.x];
  sharedTile[threadIdx.x] = perThreadSum;
  // Have the first thread compute the sum of all the elements in the
  // sharedTile.
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