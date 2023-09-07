#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

namespace kernel_4{
template<typename T>
__device__ __noinline__ void copy_to_shared(void *sharedTile, void *src, size_t nrElementsToCopy, float* temp=nullptr) {
  T* castedSharedTile = reinterpret_cast<T*>(sharedTile);
  T *castedSrc = reinterpret_cast<T *>(src);
  auto block = cooperative_groups::this_thread_block();
  auto thread = cooperative_groups::this_thread();
  // Create a pipeline.
  constexpr size_t stages_count = 1;
  auto pipeline = cuda::make_pipeline();
  for (size_t i = threadIdx.x; i < nrElementsToCopy; i += block.size())
    cuda::memcpy_async(thread, &castedSharedTile[i], &castedSrc[i], sizeof(T),
                       pipeline);
  pipeline.producer_commit();
  pipeline.consumer_wait();
  pipeline.consumer_release();
  if(threadIdx.x < 20)
    temp[threadIdx.x] = 1;
  // volatile int tempA = temp[0];
}

// Computes the sum of all the values in the array `A` and stores the result in
// the output. This kernel uses simple synchronous copy from global to shared
// memory.
template <const int BLOCK_TILE_SIZE>
__global__ void sum_with_thread_wide_memcpy_async(float *input, size_t n, float *output) {
  extern __shared__ float sharedTile[];
  __shared__ char temp[80];
  size_t allignmentInBytes = 128;
  size_t sharedTileAddr = (size_t)((void *)sharedTile);
  size_t offset = sharedTileAddr % allignmentInBytes;
  float *allignedSharedTile =
      (float *)(sharedTileAddr +
                (allignmentInBytes - offset) % allignmentInBytes);
  // Load the data from the global memory to the shared memory to be processed
  // by the current thread block.
  size_t blockStart = blockIdx.x * BLOCK_TILE_SIZE;
  size_t blockEnd = min(blockStart + BLOCK_TILE_SIZE, n);
  size_t nrElementsToCopy = blockEnd - blockStart;
  // Selects the largest possible vector-width for the copy.
  if (nrElementsToCopy % 4 == 0)
    copy_to_shared<float4>(allignedSharedTile, &input[blockStart],
                           nrElementsToCopy / 4, (float*)temp);
  else if (nrElementsToCopy % 2 == 0)
    copy_to_shared<float2>(allignedSharedTile, &input[blockStart],
                           nrElementsToCopy / 2);
  else
    copy_to_shared<float>(allignedSharedTile, &input[blockStart], nrElementsToCopy);

  // Wait for all the copies to finish.
  __syncthreads();
  // Compute the sum of all the elements in the tile.
  float perThreadSum = 0.0;
  for (size_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x)
    perThreadSum += allignedSharedTile[i - blockStart];
  __syncthreads();
  // Store the sum in the allignedSharedTile[threadId.x];
  allignedSharedTile[threadIdx.x] = perThreadSum;
  // Have the first thread compute the sum of all the elements in the
  // allignedSharedTile.
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  perThreadSum = 0.0;
  for (size_t i = 0; i < blockDim.x; i++)
    perThreadSum += allignedSharedTile[i];
  // Atomically add the result to the `output`.
  atomicAdd(output, perThreadSum);
  return;
}

} // namespace kernel_3