#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

const int N=1024;

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

int main()
{
  float *in,*out,*gold;
  float *d_in,*d_out;
  int numbytes;
  float tdiff;
  cudaEvent_t tstart,tstop;

  /* initiate timer */
  cudaEventCreate(&tstart); 
  cudaEventCreate(&tstop); 

  numbytes = N*N*sizeof(float);

  in= (float *) malloc(numbytes);
  out=(float *) malloc(numbytes);
  gold=(float *) malloc(numbytes);

  fill_matrix(in);

  cudaEventRecord(tstart); 
  transpose_cpu(in,gold);
  cudaEventRecord(tstop); 

  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);
  printf("CPU transpose: %f msec\n", tdiff);

  /* GPU section */
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);

  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

  cudaEventRecord(tstart); 
  transpose_serial<<<1,1>>>(d_in, d_out);
  cudaEventRecord(tstop); 

  cudaEventSynchronize(tstop); 
  cudaEventElapsedTime(&tdiff,tstart,tstop);

  printf("GPU serial transpose: %f msec\n", tdiff);

  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);

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

  if(compare_matrices(out,gold) == 1)
    printf("transpose FAILED!\n");
  else
    printf("transpose correct\n");

  /* print_matrix(in);*/
  /* print_matrix(out);*/

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

