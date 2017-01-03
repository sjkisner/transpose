#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

const int N=1024;
const int K=16;  /* tile size */

void transpose_cpu(float in[], float out[]);
void fill_matrix(float *in);
void print_matrix(float *in);
int compare_matrices(float *in1, float *in2);

__global__ void 
transpose_serial(float in[], float out[])
{
  int i,j;  

  for(j=0; j<N; j++)
  for(i=0; i<N; i++)
    out[j+i*N] = in[i+j*N];
}

__global__ void 
transpose_parallel_per_row(float in[], float out[])
{
  int i,j;  

  i=threadIdx.x;

  for(j=0; j<N; j++)
    out[j+i*N] = in[i+j*N];
}

__global__ void 
transpose_parallel_per_element(float in[], float out[])
{
  int i,j;  

  /* i=threadIdx.x;
     j=blockIdx.x; */

  i=threadIdx.x + (blockIdx.x * blockDim.x);
  j=threadIdx.y + (blockIdx.y * blockDim.y);

  out[j+i*N] = in[i+j*N];
}

__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
  int i,j,ii,jj;  
  __shared__ float sh_arr[K*K];

  i=threadIdx.x + (blockIdx.x * blockDim.x);
  j=threadIdx.y + (blockIdx.y * blockDim.y);

  ii=threadIdx.x;
  jj=threadIdx.y;

  /* copy+transpose current block into a tile of shared memory */
  /* read from global memory is coalesced (threadIdx.x is fast variable */
  sh_arr[jj+ii*K] = in[i+j*N];

  /* going to copy this to tile in output global array, so need to 
     complete the read+transpose */
  __syncthreads();

  /* offsets for global memory swapped here because output tile location
     is transposed position wrt input */
  /* now write to global memory is also coalesced */
  i=threadIdx.x + (blockIdx.y * blockDim.y);
  j=threadIdx.y + (blockIdx.x * blockDim.x);

  out[i+j*N] = sh_arr[ii+jj*K];

}


int main()
{
  float *h_in,*h_out,*h_gold;
  float *d_in,*d_out;
  int numbytes;
  float tdiff;
  cudaEvent_t tstart,tstop;
//  dim3 Nblocks(N,1);
  dim3 Nblocks(N/K,N/K);
  dim3 Nthreads(K,K);

  /* initiate timer */
  cudaEventCreate(&tstart); 
  cudaEventCreate(&tstop); 

  numbytes = N*N*sizeof(float);

  h_in= (float *) malloc(numbytes);
  h_out=(float *) malloc(numbytes);
  h_gold=(float *) malloc(numbytes);

  fill_matrix(h_in);

  cudaEventRecord(tstart); 
  transpose_cpu(h_in,h_gold);
  cudaEventRecord(tstop); 

  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("CPU transpose: %f msec\n", tdiff);

  /* GPU section */
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);

  cudaMemcpy(d_in, h_in,numbytes, cudaMemcpyHostToDevice);

if(0)
{
  cudaMemcpy(d_out,h_in,numbytes, cudaMemcpyHostToDevice); /* wipe first */
  cudaEventRecord(tstart); 
  transpose_serial<<<1,1>>>(d_in, d_out);
  cudaEventRecord(tstop); 
  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("GPU serial transpose: %f msec\n", tdiff);
  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);
  if(compare_matrices(h_out,h_gold) == 1) printf("transpose FAILED!\n");
}

  cudaMemcpy(d_out,h_in, numbytes, cudaMemcpyHostToDevice); /* wipe first */
  cudaEventRecord(tstart); 
  transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
  cudaEventRecord(tstop); 
  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("GPU parallel per row: %f msec\n", tdiff);
  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);
  if(compare_matrices(h_out,h_gold) == 1) printf("transpose FAILED!\n");

  cudaMemcpy(d_out,h_in, numbytes, cudaMemcpyHostToDevice); /* wipe first */
  cudaEventRecord(tstart); 
  transpose_parallel_per_element<<<Nblocks,Nthreads>>>(d_in, d_out);
  cudaEventRecord(tstop); 
  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("GPU parallel per element: %f msec\n", tdiff);
  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);
  if(compare_matrices(h_out,h_gold) == 1) printf("transpose FAILED!\n");

  cudaMemcpy(d_out,h_in, numbytes, cudaMemcpyHostToDevice); /* wipe first */
  cudaEventRecord(tstart); 
  transpose_parallel_per_element_tiled<<<Nblocks,Nthreads>>>(d_in, d_out);
  cudaEventRecord(tstop); 
  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("GPU parallel per element tiled: %f msec\n", tdiff);
  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);
  if(compare_matrices(h_out,h_gold) == 1) printf("transpose FAILED!\n");

  /* print_matrix(in);*/
  /* print_matrix(out);*/

/*
  struct timeval start,end;
  double timediff;
  gettimeofday(&start,NULL);
  // do stuff
  cudaDeviceSynchronize();   // explicit GPU barrier blocks CPU execution 
  gettimeofday(&end,NULL);   // until all device commands completed 
  timediff= (double)(end.tv_sec - start.tv_sec)*1000 +
            (double)(end.tv_usec - start.tv_usec)/1000.0;
  printf("GPU serial transpose: %f msec\n", timediff);
*/

  /* cudaDeviceReset causes the driver to clean up all state. */
  /* Calling cudaDeviceReset causes all profile data to be flushed. */
  cudaDeviceReset();
  return(0);
}


void transpose_cpu(float in[], float out[])
{
  int i,j;  

  for(j=0; j<N; j++)
  for(i=0; i<N; i++)
    out[j+i*N] = in[i+j*N];

}

void fill_matrix(float *in)
{
  int i,j;  

  for(i=0; i<N; i++)
  for(j=0; j<N; j++)
    in[i*N + j] = i*N + j;

}

void print_matrix(float *in)
{
  int i,j;  

  for(i=0; i<N; i++)
  {
    for(j=0; j<N; j++)
      printf("%d\t",(int)in[i*N + j]);
    printf("\n");
  }

}

int compare_matrices(float *in1, float *in2)
{
  int i,flag=0;  

  for(i=0; i<N*N; i++)
  if(in1[i] != in2[i])
    flag=1;

  return(flag);

}

