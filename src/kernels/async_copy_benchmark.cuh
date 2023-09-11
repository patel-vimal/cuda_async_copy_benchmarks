#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

namespace async_copy_benchmark {
__device__ void *getAlignedAddress(void *base, size_t alignmentInBytes) {
  size_t addr = (size_t)((void *)base);
  size_t offset = addr % alignmentInBytes;
  void *alignedAddr =
      (void *)(addr + (alignmentInBytes - offset) % alignmentInBytes);
  return alignedAddr;
}

// Returns the address that is aligned to `alignmentInBytes` but not to
// `alignmentInBytes*2`. For this to work correctly `alignmentInBytes` needs to
// be a power of two.
__device__ void *getAlignedAddressNotAlignedToLargerPowerOfTwo(
    void *base, size_t alignmentInBytes, size_t extraBufferMemory) {
  size_t alignedAddr = (size_t)getAlignedAddress(base, alignmentInBytes);
  // The address in `alignedAddr` can be aligned to any positive multiple of
  // 2*`alignmentInBytes`. But, note that we don't want it to be aligned to
  // 2*`alignmentInBytes`.
  if (alignedAddr % (alignmentInBytes * 2) == 0)
    alignedAddr =
        (size_t)getAlignedAddress((void *)(alignedAddr + 1), alignmentInBytes);
  size_t extraMemoryUsed = alignedAddr - size_t(base);
  if (extraMemoryUsed > extraBufferMemory)
    return nullptr;
  return (void *)alignedAddr;
}

// Copies the data from `input` GMEM buffer to the SMEM buffer using cp.async.
// This kernel should be launched with a single thread-block containing a single
// warp(32 threads). Shape and alignment of the source and destination buffers
// can be configured.
template <typename T>
__global__ void asyncCopyBenchmark(
    T *input, size_t gmemFirstDimSize, size_t gmemSecondDimSize,
    size_t gmemConsecutiveRowsPaddingElements, size_t gmemAlignmentBytes,
    size_t smemFirstDimSize, size_t smemSecondDimSize,
    size_t smemConsecutiveRowsPaddingElements, size_t smemAlignmentBytes) {
  // We'll interpret `input` as an array of shape
  // [`gmemFirstDimSize`][`gmemSecondDimSize` +
  // `gmemConsecutiveRowsPaddingElements`] and of type T.
  extern __shared__ float sharedTile[];
  T* castedSharedTile = reinterpret_cast<T*>(sharedTile);
  T *alignedSharedTile = (T *)getAlignedAddressNotAlignedToLargerPowerOfTwo(
      castedSharedTile, smemAlignmentBytes, smemAlignmentBytes * 2);
  T *alignedGmemTile = (T *)getAlignedAddressNotAlignedToLargerPowerOfTwo(
      input, gmemAlignmentBytes, gmemAlignmentBytes * 2);
  if (!alignedSharedTile || !alignedGmemTile)
    return;
  // Thread with threadId.x=i will copy the buffer element with indices
  // [i/gmemSecondDimSize][i%gmemSecondDimSize].
  auto thread = cooperative_groups::this_thread();
  auto pipeline = cuda::make_pipeline();
  size_t tId = threadIdx.x;
  size_t gmemIndex = (tId / gmemSecondDimSize) *
                         (gmemSecondDimSize + gmemConsecutiveRowsPaddingElements) +
                     (tId % smemSecondDimSize);
  size_t smemIndex = (tId / smemSecondDimSize) *
                         (smemSecondDimSize + smemConsecutiveRowsPaddingElements) +
                     (tId % smemSecondDimSize);
  pipeline.producer_acquire();
  cuda::memcpy_async(thread, alignedSharedTile + smemIndex,
                     alignedGmemTile + gmemIndex, sizeof(T), pipeline);
  pipeline.producer_commit();
  pipeline.consumer_wait();
  pipeline.consumer_release();
  __syncthreads();
  return;
}

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

void asyncCopyBenchmarkRun(size_t gmemFirstDimSize, size_t gmemSecondDimSize,
                           size_t gmemConsecutiveRowsPaddingBytes,
                           size_t gmemAlignmentBytes, size_t smemFirstDimSize,
                           size_t smemSecondDimSize,
                           size_t smemConsecutiveRowsPaddingBytes,
                           size_t smemAlignmentBytes, size_t vectorSize) {
  assert(vectorSize == 1 || vectorSize == 2 || vectorSize == 4);
  assert(gmemConsecutiveRowsPaddingBytes % (vectorSize * 4) == 0);
  assert(smemConsecutiveRowsPaddingBytes % (vectorSize * 4) == 0);
  size_t gmemConsecutiveRowsPaddingElements =
      gmemConsecutiveRowsPaddingBytes / (vectorSize * 4);
  size_t smemConsecutiveRowsPaddingElements =
      smemConsecutiveRowsPaddingBytes / (vectorSize * 4);
  size_t gmemRequired =
      gmemFirstDimSize *
      (gmemSecondDimSize + gmemConsecutiveRowsPaddingElements) * vectorSize * 4;
  gmemRequired += gmemAlignmentBytes * 2;
  size_t smemRequired =
      smemFirstDimSize *
      (smemSecondDimSize + smemConsecutiveRowsPaddingElements) * vectorSize * 4;
  smemRequired += smemAlignmentBytes * 2;
  float *devicePtr;
  gpuErrchk(cudaMalloc((void **)&devicePtr, gmemRequired));
  switch (vectorSize) {
  case 1:
    asyncCopyBenchmark<float><<<1, 32, smemRequired>>>(
        (float *)devicePtr, gmemFirstDimSize, gmemSecondDimSize,
        gmemConsecutiveRowsPaddingElements, gmemAlignmentBytes,
        smemFirstDimSize, smemSecondDimSize, smemConsecutiveRowsPaddingElements,
        smemAlignmentBytes);
    break;
  case 2:
    asyncCopyBenchmark<float2><<<1, 32, smemRequired>>>(
        (float2 *)devicePtr, gmemFirstDimSize, gmemSecondDimSize,
        gmemConsecutiveRowsPaddingElements, gmemAlignmentBytes,
        smemFirstDimSize, smemSecondDimSize, smemConsecutiveRowsPaddingElements,
        smemAlignmentBytes);
    break;
  case 4:
    asyncCopyBenchmark<float4><<<1, 32, smemRequired>>>(
        (float4 *)devicePtr, gmemFirstDimSize, gmemSecondDimSize,
        gmemConsecutiveRowsPaddingElements, gmemAlignmentBytes,
        smemFirstDimSize, smemSecondDimSize, smemConsecutiveRowsPaddingElements,
        smemAlignmentBytes);
    break;
  }
  return;
}

} // namespace async_copy_benchmark
