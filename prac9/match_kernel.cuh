#include <helper_cuda.h>


///////////////////////////////////////////////////////////////////////
// GPU routine
///////////////////////////////////////////////////////////////////////

__global__ void match_kernel(int *matches, const unsigned int* text, const unsigned int* words, const int length, const int nwords)
{
  // Dynamically allocated shared memory for scan kernels

  // __shared__ unsigned int stext[16]; // 4 * nwords
  extern __shared__  unsigned int sdata[];
  unsigned int *scnts = sdata + blockDim.x * 4;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int wid = threadIdx.y;

  if (tid < length && wid < 4) {
    sdata[4 * threadIdx.x + wid] = (text[tid]>>(8*wid)) + (text[tid+1]<<(32-8*wid)); 
  }
  __syncthreads();

  if (tid < length) {
    for (int i=0; i<4; ++i) {
      if (sdata[4 * threadIdx.x + i] == words[wid]) {
        atomicAdd(scnts+wid, 1);
      }
    }
  }
  __syncthreads();

  if (tid < length && threadIdx.x == 0 && threadIdx.y < 4) {
    atomicAdd(matches+threadIdx.y, scnts[threadIdx.y]);
  }
}