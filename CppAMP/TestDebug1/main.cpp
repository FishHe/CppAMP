#include <stdio.h>  
#include <tchar.h> 

#include <amp.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace Concurrency;

//the size of matrix
#define N 512

void main()
{
	// the arrray of matrix A
	float *A = new float[N*N];
	for (int i = 0; i < N*N; i++) A[i] = 1.0f;
	// the arrray of matrix A
	float *B = new float[N*N];
	for (int i = 0; i < N*N; i++) B[i] = 1.0f;
	// the array of product matrix
	float *P = new float[N*N];
	for (int i = 0; i < N*N; i++) P[i] = 0.0f;
}