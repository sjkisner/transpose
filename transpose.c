#include <stdio.h>
#include <stdlib.h>

const int N=10;

void transpose_cpu(float in[], float out[]);
void fill_matrix(float *in);
void print_matrix(float *in);


int main()
{
  float *in,*out;

  in= (float *) malloc(N*N*sizeof(float));
  out=(float *) malloc(N*N*sizeof(float));

  fill_matrix(in);
  transpose_cpu(in,out);

  print_matrix(in);
  printf("\n");
  print_matrix(out);
  printf("\n");
  printf("done\n");

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

