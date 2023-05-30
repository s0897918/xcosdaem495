#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

int checkGPU() {
  int deviceCount;
  hipError_t error_id = hipGetDeviceCount(&deviceCount);
  if (error_id != hipSuccess) {
    printf("hipGetDeviceCount returned %d\n-> %s\n", error_id, hipGetErrorString(error_id));
  }

  printf("%d\n", deviceCount);
  if (deviceCount == 0) {
    printf("There are no available device(s) that support ROCm\n");
  } else {
    printf("Detected %d ROCm Capable device(s)\n", deviceCount);
  }
  return deviceCount;
}


__global__ void kernelT(float *p)
{
  int tid = threadIdx.x;
  p[tid] = tid;
}

void runCUDATest(){
  float *h_p, *d_p;
  int size = 16;
  h_p = (float *)malloc(size * sizeof(float));
  hipMalloc((void **)&d_p, size*sizeof(float));
  //dim3 dimGrid (1, 1, 1);
  //dim3 dimBlock (256, 1, 1);
  kernelT <<<1, 16>>>(d_p);
  hipDeviceSynchronize();

  hipMemcpy(h_p,d_p,size * sizeof(float),hipMemcpyDeviceToHost);
  for (int i=0; i<size; i++) printf("%f ", h_p[i]);
  hipFree(d_p);
}


int main(){

  int deviceCount = checkGPU();
 runCUDATest();

}